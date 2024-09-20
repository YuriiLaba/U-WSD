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
            warmup_steps = int(0.1 * total_steps)
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

    def train_step(self, batch):
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
                    eval_loss += self.train_step(eval_batch)
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
            loss = self.train_step(batch)
            loss.backward()

            self.train_avg_meter.update(loss.item(), self.config.getint('MODEL_TUNING', 'batch_size'))  # TODO: .detach().cpu()?
            
            self.optim.step()
            self.optim.zero_grad()

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
    

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# def _read_wsd_eval_dataset(config):
#     wsd_eval_data = pd.read_csv(config["MODEL_TUNING"]["path_to_wsd_eval_dataset"])
#     wsd_eval_data["examples"] = wsd_eval_data["examples"].apply(lambda x: literal_eval(x))
#     wsd_eval_data["gloss"] = wsd_eval_data["gloss"].apply(lambda x: literal_eval(x))
#     return wsd_eval_data


# def _load_dataset(config):
#     data = pd.read_csv(config["MODEL_TUNING"]["path_to_triplet_dataset"])
#     data = data.sample(frac=1)
#     train_data = data[:int(len(data)*0.99)]
#     eval_data = data[int(len(data)*0.99):]
#     return train_data, eval_data


# def _load_model_and_tokenizer(config):
#     tokenizer = AutoTokenizer.from_pretrained(config["MODEL_TUNING"]["model_to_fine_tune"])
#     model = AutoModel.from_pretrained(config["MODEL_TUNING"]["model_to_fine_tune"],
#                                       output_hidden_states=True).to(device)
#     return model, tokenizer


# def _mean_pool(token_embeds, attention_mask):
#     in_mask = attention_mask.unsqueeze(-1).expand(
#         token_embeds.size()
#     ).float()

#     pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
#         in_mask.sum(1), min=1e-9
#     )
#     return pool


# def _calculate_triplet_loss(model, batch, triplet_loss):
#     anchor_ids = batch['anchor_ids'].to(device)
#     anchor_mask = batch['anchor_mask'].to(device)
#     pos_ids = batch['positive_ids'].to(device)
#     pos_mask = batch['positive_mask'].to(device)
#     neg_ids = batch['negative_ids'].to(device)
#     neg_mask = batch['negative_mask'].to(device)

#     a = model(anchor_ids, attention_mask=anchor_mask)[0]
#     p = model(pos_ids, attention_mask=pos_mask)[0]
#     n = model(neg_ids, attention_mask=neg_mask)[0]

#     a = _mean_pool(a, anchor_mask)
#     p = _mean_pool(p, pos_mask)
#     n = _mean_pool(n, neg_mask)

#     del anchor_ids, anchor_mask, pos_ids, pos_mask, neg_ids, neg_mask, batch

#     return triplet_loss(a, p, n)


# def _calculate_wsd_accuracy(model, udpipe_model, eval_data, tokenizer):
#     word_sense_detector = WordSenseDetector(
#         pretrained_model=model,
#         udpipe_model=udpipe_model,
#         evaluation_dataset=eval_data,
#         tokenizer=tokenizer,
#         pooling_strategy=PoolingStrategy.mean_pooling,
#         prediction_strategy=PredictionStrategy.all_examples_to_one_embedding
#     )

#     eval_data = word_sense_detector.run()
#     return prediction_accuracy(eval_data)

# def _save_model(model, path_to_save_model):
#     if isinstance(model, torch.nn.DataParallel):
#         model.module.save_pretrained(path_to_save_model, from_pt=True)
#     else:
#         model.save_pretrained(path_to_save_model, from_pt=True)


# def train(config):

#     if config.getboolean("MODEL_TUNING", "log_to_neptune"):
#         # TODO: move to config
#         run = neptune.init_run(
#             project="vova.mudruy/WSD",
#             api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==",
#         )

#     batch_count = 0
#     rounds_count = 0
#     max_wsd_acc = 0

#     udpipe_model = UDPipeModel(config["MODEL_TUNING"]["path_to_udpipe_model"])

#     triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

#     if not os.path.isdir(config["MODEL_TUNING"]["path_to_save_fine_tuned_model"]):
#         os.mkdir(config["MODEL_TUNING"]["path_to_save_fine_tuned_model"])

#     wsd_eval_data = _read_wsd_eval_dataset(config)
#     train_data, eval_data = _load_dataset(config)
#     model, tokenizer = _load_model_and_tokenizer(config)

#     model = nn.DataParallel(model)

#     if config.getboolean('MODEL_TUNING', 'random_model_weights_reinitialization'):
#         model = ModelWithRandomizingSomeWeights(model=model,
#                                                 reinit_n_layers=config.getint('MODEL_TUNING',
#                                                                               'number_of_layers_for_reinitialization')).to(
#             device)

#     train_dataset = TripletDataset(anchor=train_data["anchor"].values,
#                                    positive=train_data["positive"].values,
#                                    negative=train_data["negative"].values,
#                                    tokenizer=tokenizer)

#     eval_dataset = TripletDataset(anchor=eval_data["anchor"].values,
#                                    positive=eval_data["positive"].values,
#                                    negative=eval_data["negative"].values,
#                                    tokenizer=tokenizer)
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=config.getint('MODEL_TUNING', 'batch_size'), shuffle=True,
#                                                num_workers=4, pin_memory=True)
#     eval_loader = torch.utils.data.DataLoader(eval_dataset,
#                                               batch_size=config.getint('MODEL_TUNING', 'batch_size'), shuffle=True,
#                                               num_workers=4, pin_memory=True)

#     train_loss_stat = AverageMeter("train_loss")
#     # eval_loss_stat = AverageMeter("eval_loss")

#     optim = torch.optim.Adam(model.parameters(), lr=config.getfloat('MODEL_TUNING', 'learning_rate'))

#     if config.getboolean('MODEL_TUNING', 'apply_warmup'):
#         total_steps = int(len(train_dataset) / config.getint('MODEL_TUNING', 'batch_size'))
#         warmup_steps = int(0.1 * total_steps)
#         scheduler = get_linear_schedule_with_warmup(
#             optim, num_warmup_steps=warmup_steps,
#             num_training_steps=total_steps - warmup_steps
#         )
#     if config.getboolean("MODEL_TUNING", "log_to_neptune"):
#         run['epochs'] = config.getint('MODEL_TUNING', 'num_epochs')
#         run['batch_size'] = config.getint('MODEL_TUNING', 'batch_size')
#         run["learning_rate"] = config.getfloat('MODEL_TUNING', 'learning_rate')
#         run["early_stopping"] = config.getint('MODEL_TUNING', 'early_stopping')
#         run["dataset/train"] = len(train_dataset)
#         run["dataset/eval"] = len(eval_dataset)
#         run["dataset/diff_threshold"] = 0.3

#     for epoch in range(config.getint('MODEL_TUNING', 'num_epochs')):
#         model.train()
#         loop = tqdm(train_loader, leave=True)

#         for batch in loop:
#             if config["MODEL_TUNING"]["loss"] == "mnr_loss":
#                 pass
#                 # loss = calculate_mnr_loss(model, batch)
#             elif config["MODEL_TUNING"]["loss"] == "triplet_loss":
#                 loss = _calculate_triplet_loss(model, batch, triplet_loss)
#             else:
#                 raise Exception("Undefined loss!")

#             train_loss_stat.update(loss.item(), config.getint('MODEL_TUNING', 'batch_size'))  # TODO: .detach().cpu()?

#             loss.backward()
#             optim.step()
#             optim.zero_grad()

#             if config.getboolean('MODEL_TUNING', 'apply_warmup'):
#                 scheduler.step()

#             if config.getboolean("MODEL_TUNING", "log_to_neptune"):
#                 acc_val, acc_avg = train_loss_stat()
#                 run["train/loss"].append(acc_avg)
#             report_gpu()

#             if batch_count % 100 == 0:
#                 model.eval()
#                 eval_loss = 0

#                 with torch.no_grad():
#                     eval_loop = tqdm(eval_loader, leave=True)
#                     for eval_batch in eval_loop:
#                         with torch.cuda.amp.autocast():  # TODO: Do we need this?
#                             if config["MODEL_TUNING"]["loss"] == "mnr":
#                                 pass
#                                 # eval_loss += calculate_mnr_loss(model, eval_batch).item()
#                             if config["MODEL_TUNING"]["loss"] == "triplet_loss":
#                                 eval_loss += _calculate_triplet_loss(model, eval_batch, triplet_loss).item()
#                         report_gpu()

#                     print("Eval loss: " + str(round(eval_loss / len(eval_loader), 3)))

#                     wsd_acc = _calculate_wsd_accuracy(model, udpipe_model, wsd_eval_data, tokenizer)
#                     report_gpu()
#                     print("WSD accuracy: " + str(wsd_acc))

#                     if config.getboolean("MODEL_TUNING", "log_to_neptune"):
#                         run["eval/wsd_acc"].append(wsd_acc)
#                         run["eval/loss"].append(eval_loss / len(eval_loader))

#                     if wsd_acc > max_wsd_acc:
#                         max_wsd_acc = wsd_acc
#                         rounds_count = 0
#                         try:
#                             if batch_count > 0:
#                                 # TODO: model won't be save if neptune is false
#                                 _save_model(model, f"{config['MODEL_TUNING']['path_to_save_fine_tuned_model']}/model_{run['sys/id'].fetch().split('/')[-1][4:]}_{epoch}")
#                         except Exception as e:
#                             print(f'model not saved epoch = {epoch}, batch = {batch_count}, error = {e}')

#                     elif wsd_acc < max_wsd_acc:
#                         rounds_count += 1

#                     if rounds_count == config.getint('MODEL_TUNING', 'early_stopping'):
#                         print(f'Early stopping, model not improve WSD for {config.getint("MODEL_TUNING", "early_stopping")}')
#                         return
#                 model.train()

#             batch_count += 1
#             loop.set_description(f'Epoch {epoch}')

#         # epoch ended
#         try:
#             # TODO: model won't be save if neptune is false
#             _save_model(model, f"{config['MODEL_TUNING']['path_to_save_fine_tuned_model']}/model_{run['sys/id'].fetch().split('/')[-1][4:]}_{epoch}")
#         except Exception as e:
#             print(f'model not saved epoch = {epoch}, batch = {batch_count}, error = {e}')
#         batch_count = 0

#     if config.getboolean("MODEL_TUNING", "log_to_neptune"):
#         run.stop()
