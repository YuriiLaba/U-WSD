import os
import sys
sys.path.append("src/model_fine_tuning")

import pandas as pd
from ast import literal_eval
import torch
import numpy as np
import random

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
from sklearn.metrics import accuracy_score

import neptune.new as neptune
from tqdm.auto import tqdm

from reinit_model_weights import ModelWithRandomizingSomeWeights
from triplet_dataset import TripletDataset
from src.udpipe_model import UDPipeModel
from src.word_sense_detector import WordSenseDetector
from utlis import report_gpu

torch.manual_seed(47)
random.seed(92)
np.random.seed(39)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _read_wsd_eval_dataset(config):
    wsd_eval_data = pd.read_csv(config["MODEL_TUNING"]["path_to_wsd_eval_dataset"])
    wsd_eval_data["examples"] = wsd_eval_data["examples"].apply(lambda x: literal_eval(x))
    wsd_eval_data["gloss"] = wsd_eval_data["gloss"].apply(lambda x: literal_eval(x))
    return wsd_eval_data


def _load_dataset(config):
    data = pd.read_csv(config["MODEL_TUNING"]["path_to_triplet_dataset"])
    data = data.sample(frac=1)
    train_data = data[:int(len(data)*0.99)]
    eval_data = data[int(len(data)*0.99):]
    return train_data, eval_data


def _load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_TUNING"]["model_to_fine_tune"])
    model = AutoModel.from_pretrained(config["MODEL_TUNING"]["model_to_fine_tune"],
                                      output_hidden_states=True).to(device)
    return model, tokenizer


def _mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()

    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool


def _calculate_triplet_loss(model, batch, triplet_loss):
    anchor_ids = batch['anchor_ids'].to(device)
    anchor_mask = batch['anchor_mask'].to(device)
    pos_ids = batch['positive_ids'].to(device)
    pos_mask = batch['positive_mask'].to(device)
    neg_ids = batch['negative_ids'].to(device)
    neg_mask = batch['negative_mask'].to(device)

    a = model(anchor_ids, attention_mask=anchor_mask)[0]
    p = model(pos_ids, attention_mask=pos_mask)[0]
    n = model(neg_ids, attention_mask=neg_mask)[0]

    a = _mean_pool(a, anchor_mask)
    p = _mean_pool(p, pos_mask)
    n = _mean_pool(n, neg_mask)

    del anchor_ids, anchor_mask, pos_ids, pos_mask, neg_ids, neg_mask, batch

    return triplet_loss(a, p, n)


# TODO: REMOVE THIS!!
def prediction_accuracy(data_with_predictions):
    data_dropna = data_with_predictions.dropna()
    data_dropna['gloss'] = data_dropna['gloss'].apply(lambda x: x[0])
    data_dropna['predicted_context'] = data_dropna['predicted_context'].apply(lambda x: x[0])
    return accuracy_score(data_dropna["gloss"], data_dropna["predicted_context"])


def _calculate_wsd_accuracy(model, udpipe_model, eval_data, tokenizer):
    word_sense_detector = WordSenseDetector(
        pretrained_model=model,
        udpipe_model=udpipe_model,
        evaluation_dataset=eval_data,
        tokenizer=tokenizer)

    eval_data = word_sense_detector.run()
    return prediction_accuracy(eval_data)


def train(config):

    if config.getboolean("MODEL_TUNING", "log_to_neptune"):
        # TODO: move to config
        run = neptune.init_run(
            project="vova.mudruy/WSD",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==",
        )

    batch_count = 0
    rounds_count = 0
    max_wsd_acc = 0

    udpipe_model = UDPipeModel(config["MODEL_TUNING"]["path_to_udpipe_model"])

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    if not os.path.isdir(config["MODEL_TUNING"]["path_to_save_fine_tuned_model"]):
        os.mkdir(config["MODEL_TUNING"]["path_to_save_fine_tuned_model"])

    wsd_eval_data = _read_wsd_eval_dataset(config)
    train_data, eval_data = _load_dataset(config)
    model, tokenizer = _load_model_and_tokenizer(config)

    if config.getboolean('MODEL_TUNING', 'random_model_weights_reinitialization'):
        model = ModelWithRandomizingSomeWeights(model=model,
                                                reinit_n_layers=config.getint('MODEL_TUNING',
                                                                              'number_of_layers_for_reinitialization')).to(
            device)

    train_dataset = TripletDataset(anchor=train_data["anchor"].values,
                                   positive=train_data["positive"].values,
                                   negative=train_data["negative"].values,
                                   tokenizer=tokenizer)

    eval_dataset = TripletDataset(anchor=eval_data["anchor"].values,
                                   positive=eval_data["positive"].values,
                                   negative=eval_data["negative"].values,
                                   tokenizer=tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.getint('MODEL_TUNING', 'batch_size'), shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=config.getint('MODEL_TUNING', 'batch_size'), shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=config.getfloat('MODEL_TUNING', 'learning_rate'))

    if config.getboolean('MODEL_TUNING', 'apply_warmup'):
        total_steps = int(len(train_dataset) / config.getint('MODEL_TUNING', 'batch_size'))
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps - warmup_steps
        )
    if config.getboolean("MODEL_TUNING", "log_to_neptune"):
        run['epochs'] = config.getint('MODEL_TUNING', 'num_epochs')
        run['batch_size'] = config.getint('MODEL_TUNING', 'batch_size')
        run["learning_rate"] = config.getfloat('MODEL_TUNING', 'learning_rate')
        run["early_stopping"] = config.getint('MODEL_TUNING', 'early_stopping')
        run["dataset/train"] = len(train_dataset)
        run["dataset/eval"] = len(eval_dataset)
        run["dataset/diff_threshold"] = 0.3

    for epoch in range(config.getint('MODEL_TUNING', 'num_epochs')):
        model.train()
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            if config["MODEL_TUNING"]["loss"] == "mnr_loss":
                pass
                # loss = calculate_mnr_loss(model, batch)
            elif config["MODEL_TUNING"]["loss"] == "triplet_loss":
                loss = _calculate_triplet_loss(model, batch, triplet_loss)
            else:
                raise Exception("Undefined loss!")

            loss.backward()
            optim.step()
            optim.zero_grad()

            if config.getboolean('MODEL_TUNING', 'apply_warmup'):
                scheduler.step()

            if config.getboolean("MODEL_TUNING", "log_to_neptune"):
                run["train/loss"].append(loss.item())
            report_gpu()

            if batch_count % 100 == 0:
                model.eval()
                eval_loss = 0

                with torch.no_grad():
                    for eval_batch in eval_loader:
                        with torch.cuda.amp.autocast():
                            if config["MODEL_TUNING"]["loss"] == "mnr":
                                pass
                                # eval_loss += calculate_mnr_loss(model, eval_batch).item()
                            if config["MODEL_TUNING"]["loss"] == "triplet_loss":
                                eval_loss += _calculate_triplet_loss(model, eval_batch, triplet_loss).item()
                        report_gpu()

                    print("Eval loss: " + str(round(eval_loss / len(eval_loader), 3)))

                    wsd_acc = _calculate_wsd_accuracy(model, udpipe_model, wsd_eval_data, tokenizer)
                    report_gpu()
                    print("WSD accuracy: " + str(wsd_acc))

                    if config.getboolean("MODEL_TUNING", "log_to_neptune"):
                        run["eval/wsd_acc"].append(wsd_acc)
                        run["eval/loss"].append(eval_loss / len(eval_loader))

                    if wsd_acc > max_wsd_acc:
                        max_wsd_acc = wsd_acc
                        rounds_count = 0
                        try:
                            if batch_count > 0:
                                model.save_pretrained(f"{config['MODEL_TUNING']['path_to_save_fine_tuned_model']}/model_{run.get_run_url().split('/')[-1][4:]}_{epoch}_{batch_count}", from_pt=True)
                        except Exception as e:
                            print(f'model not saved epoch = {epoch}, batch = {batch_count}, error = {e}')

                    elif wsd_acc < max_wsd_acc:
                        rounds_count += 1

                    if rounds_count == config.getint('MODEL_TUNING', 'early_stopping'):
                        print(f'Early stopping, model not improve WSD for {config.getint("MODEL_TUNING", "early_stopping")}')
                        return
                model.train()

            batch_count += 1
            loop.set_description(f'Epoch {epoch}')

        # epoch ended
        try:
            # TODO: model won't be save if neptune is false
            model.save_pretrained(f"{config['MODEL_TUNING']['path_to_save_fine_tuned_model']}/model_{run.get_run_url().split('/')[-1][4:]}_{epoch}", from_pt=True)
        except Exception as e:
            print(f'model not saved epoch = {epoch}, batch = {batch_count}, error = {e}')
        batch_count = 0

    if config.getboolean("MODEL_TUNING", "log_to_neptune"):
        run.stop()
