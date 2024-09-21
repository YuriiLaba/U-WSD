import os
import sys

import pandas as pd
from ast import literal_eval
import torch
import numpy as np
import random

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn

import neptune.new as neptune
from tqdm.auto import tqdm

from services.trainer.reinit_model_weights import ModelWithRandomizingSomeWeights
from services.trainer.triplet_dataset import TripletDataset
from services.udpipe_model import UDPipeModel
from services.word_sense_detector import WordSenseDetector
from services.utils_results import prediction_accuracy
from services.poolings import PoolingStrategy
from services.prediction_strategies import PredictionStrategy
from services.trainer.utlis import report_gpu
from services.trainer.utlis import AverageMeter

import warnings
warnings.simplefilter('ignore')

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(47)
random.seed(92)
np.random.seed(39)

# TODO: add logging to the class Trainer
class Trainer:
    def __init__(self, config):
        # TODO: i think that a lot of the following code should be move to separate file
        self.config = config
        self.log_to_neptune = self.config.getboolean("MODEL_TUNING", "log_to_neptune")
        self.apply_warmup = self.config.getboolean('MODEL_TUNING', 'apply_warmup')

        if self.log_to_neptune:
            # TODO: move to config
            self.run = neptune.init_run(
                project="vova.mudruy/WSD",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==",
            )
            self.run['epochs'] = config.getint('MODEL_TUNING', 'num_epochs')
            self.run['batch_size'] = config.getint('MODEL_TUNING', 'batch_size')
            self.run["learning_rate"] = config.getfloat('MODEL_TUNING', 'learning_rate')
            self.run["early_stopping"] = config.getint('MODEL_TUNING', 'early_stopping')
            self.run["dataset/diff_threshold"] = 0.3
            
        
        self.udpipe_model = UDPipeModel(self.config["MODEL_TUNING"]["path_to_udpipe_model"])
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        if not os.path.isdir(self.config["MODEL_TUNING"]["path_to_save_fine_tuned_model"]):
            os.mkdir(self.config["MODEL_TUNING"]["path_to_save_fine_tuned_model"])
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_TUNING"]["model_to_fine_tune"])
        self.model = AutoModel.from_pretrained(self.config["MODEL_TUNING"]["model_to_fine_tune"], 
                                          output_hidden_states=True).to(self.device)
        
        if self.config.getboolean('MODEL_TUNING', 'enable_gpu_parallel'):
            self.model = nn.DataParallel(self.model)
        
        if self.config.getboolean('MODEL_TUNING', 'random_model_weights_reinitialization'): # TODO: for now we don't apply it
            self.model = ModelWithRandomizingSomeWeights(model=self.model, 
                                                         reinit_n_layers=self.config.getint('MODEL_TUNING',
                                                         'number_of_layers_for_reinitialization')).to(self.device)
        
        self.wsd_eval_data = self._load_wsd_eval_dataset()
        self.train_data, self.eval_data = self._load_train_eval_datasets()

        self.train_dataset = TripletDataset(anchor=self.train_data["anchor"].values,
                                   positive=self.train_data["positive"].values,
                                   negative=self.train_data["negative"].values,
                                   tokenizer=self.tokenizer)

        self.eval_dataset = TripletDataset(anchor=self.eval_data["anchor"].values,
                                   positive=self.eval_data["positive"].values,
                                   negative=self.eval_data["negative"].values,
                                   tokenizer=self.tokenizer)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                               batch_size=config.getint('MODEL_TUNING', 'batch_size'), shuffle=True,
                                               num_workers=4, pin_memory=True)
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset,
                                              batch_size=config.getint('MODEL_TUNING', 'batch_size'), shuffle=True,
                                              num_workers=4, pin_memory=True)

        self.train_avg_meter = AverageMeter("train_loss")
        self.max_wsd_acc = 0
        self.rounds_count = 0

        # TODO: i'd like to have better config parsing. It takes too much space 
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.getfloat('MODEL_TUNING', 'learning_rate'))

        if self.apply_warmup:
            total_steps = int(len(self.train_dataset) / config.getint('MODEL_TUNING', 'batch_size'))
            warmup_steps = int(config.getfloat('MODEL_TUNING', 'warmup_ration') * total_steps)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optim, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps - warmup_steps
            )
        
        if self.log_to_neptune:
            self.run["dataset/train"] = len(self.train_data)
            self.run["dataset/eval"] = len(self.eval_data)
    
    def _load_wsd_eval_dataset(self):
        wsd_eval_data = pd.read_csv(self.config["MODEL_TUNING"]["path_to_wsd_eval_dataset"])
        wsd_eval_data["examples"] = wsd_eval_data["examples"].apply(lambda x: literal_eval(x))
        wsd_eval_data["gloss"] = wsd_eval_data["gloss"].apply(lambda x: literal_eval(x))
        return wsd_eval_data

    def _load_train_eval_datasets(self):
        data = pd.read_csv(self.config["MODEL_TUNING"]["path_to_triplet_dataset"])
        data = data.sample(frac=1)
        train_data = data[:int(len(data)*0.99)]
        eval_data = data[int(len(data)*0.99):]
        return train_data, eval_data
    
    def _batch_mean_pooling(self, token_embeds, attention_mask):
        # TODO: analyze this method. Maybe it can be moved/merged in Pooling
        in_mask = attention_mask.unsqueeze(-1).expand(
            token_embeds.size()
        ).float()

        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-9
        )
        return pool
    
    def _calculate_triplet_loss(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        a = self.model(batch['anchor_ids'], attention_mask=batch['anchor_mask'])[0]
        p = self.model(batch['positive_ids'], attention_mask=batch['positive_mask'])[0]
        n = self.model(batch['negative_ids'], attention_mask=batch['negative_mask'])[0]

        a = self._batch_mean_pooling(a, batch['anchor_mask'])
        p = self._batch_mean_pooling(p, batch['positive_mask'])
        n = self._batch_mean_pooling(n, batch['negative_mask'])

        # del batch
        return self.triplet_loss(a, p, n)
    
    def _calculate_wsd_accuracy(self, eval_data):
        word_sense_detector = WordSenseDetector(
            pretrained_model=self.model,
            udpipe_model=self.udpipe_model,
            evaluation_dataset=eval_data,
            tokenizer=self.tokenizer,
            pooling_strategy=PoolingStrategy.mean_pooling,
            prediction_strategy=PredictionStrategy.all_examples_to_one_embedding
        )

        eval_data = word_sense_detector.run()
        return prediction_accuracy(eval_data)

    def _save_model(self, model, path_to_save_model):
        try:
            if isinstance(model, torch.nn.DataParallel):
                model.module.save_pretrained(path_to_save_model, from_pt=True)
            else:
                model.save_pretrained(path_to_save_model, from_pt=True)
        except Exception as e:
            print(f'model not saved, error = {e}')

    def calculate_loss(self, batch):
        loss_type = self.config["MODEL_TUNING"]["loss"]
        
        if loss_type == "triplet_loss": # TODO: we also wanted to add mnr loss
            return self._calculate_triplet_loss(batch)       
        raise Exception(f"Undefined loss: {loss_type}")
    
    def evaluate_epoch(self, epoch, batch_count):
        self.model.eval()
        eval_loss = 0

        with torch.no_grad():
            eval_bar = tqdm(self.eval_loader, leave=True, desc='Triplet Eval')
            for eval_batch in eval_bar:
                with torch.cuda.amp.autocast():  # TODO: Do we need this?
                    eval_loss += self.calculate_loss(eval_batch)
                report_gpu()

            if self.log_to_neptune:
                self.run["eval/loss"].append(eval_loss / len(self.eval_loader))

            wsd_acc = self._calculate_wsd_accuracy(self.wsd_eval_data)
            report_gpu()

            if self.log_to_neptune:
                self.run["eval/wsd_acc"].append(wsd_acc)

            if wsd_acc > self.max_wsd_acc:
                self.max_wsd_acc = wsd_acc
                self.rounds_count = 0
            
                if batch_count > 0:
                    # TODO: model won't be save if neptune is false            
                    self._save_model(self.model, f"{self.config['MODEL_TUNING']['path_to_save_fine_tuned_model']}/model_{self.run['sys/id'].fetch().split('/')[-1][4:]}_{epoch}")

            elif wsd_acc < self.max_wsd_acc:
                self.rounds_count += 1

            if self.rounds_count == self.config.getint('MODEL_TUNING', 'early_stopping'):
                print(f'Early stopping, model not improve WSD for {self.config.getint("MODEL_TUNING", "early_stopping")}')
                return True
        return False

    def train_epoch(self, epoch):
        train_bar = tqdm(self.train_loader, leave=True, desc=f'Train epoch: {epoch}')

        for batch_count, batch in enumerate(train_bar):
            self.model.train()

            loss = self.calculate_loss(batch)
            
            self.optim.zero_grad()
            loss.backward()
            
            self.optim.step()
            self.train_avg_meter.update(loss.item(), self.config.getint('MODEL_TUNING', 'batch_size'))  # TODO: .detach().cpu()?

            if self.apply_warmup:
                self.scheduler.step()

            if self.log_to_neptune:
                acc_val, acc_avg = self.train_avg_meter()
                self.run["train/loss"].append(acc_avg)

            report_gpu() # TODO: it's interesting do we really need it

            if batch_count % self.config.getint('MODEL_TUNING', 'num_batch_to_eval') == 0 and batch_count != 0:
                if self.evaluate_epoch(epoch, batch_count):
                    return # reach early stopping rounds
    
    def train(self):
        # initial evaluation of the raw model
        self.evaluate_epoch(epoch=0, batch_count=0)

        for epoch in range(self.config.getint('MODEL_TUNING', 'num_epochs')):
            self.train_epoch(epoch)
            # TODO: model won't be save if neptune is false because of run
            self._save_model(self.model, f"{self.config['MODEL_TUNING']['path_to_save_fine_tuned_model']}/model_{self.run['sys/id'].fetch().split('/')[-1][4:]}_{epoch}")

        if self.log_to_neptune:
            self.run.stop()
