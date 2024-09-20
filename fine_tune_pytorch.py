from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from transformers.optimization import get_linear_schedule_with_warmup
import neptune.new as neptune
from tqdm.auto import tqdm
import pandas as pd
from ast import literal_eval
import os
import gc
import torch.nn as nn

from services.word_sense_detector import WordSenseDetector
from services.poolings import PoolingStrategy
from services.prediction_strategies import PredictionStrategy
from services.udpipe_model import UDPipeModel
from sklearn.metrics import accuracy_score

import torch
import random
import numpy as np
import transformers

torch.manual_seed(47)
random.seed(92)
np.random.seed(39)


def report_gpu():
   torch.cuda.empty_cache()
   gc.collect()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cos_sim = torch.nn.CosineSimilarity().to(device)
cross_entropy_loss = torch.nn.CrossEntropyLoss().to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)


def roberta_base_AdamW_LLRD(model):

    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = 3.5e-6
    head_lr = 3.6e-6
    lr = init_lr

    # === Pooler and regressor ======================================================

    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}
    opt_parameters.append(head_params)

    # === 12 Hidden layers ==========================================================

    for layer in range(11,-1,-1):
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)

        lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n,p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
    opt_parameters.append(embed_params)

    return transformers.AdamW(opt_parameters, lr=init_lr)

class ModelWithRandomizingSomeWeights(nn.Module):

    def __init__(self, reinit_n_layers=0):
        super().__init__()
        self.roberta_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', output_hidden_states=True)
        # self.regressor = nn.Linear(768, 1)
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit()

    def _do_reinit(self):
        # Re-init pooler.
        self.roberta_model.pooler.dense.weight.data.normal_(mean=0.0, std=self.roberta_model.config.initializer_range)
        self.roberta_model.pooler.dense.bias.data.zero_()
        for param in self.roberta_model.pooler.parameters():
            param.requires_grad = True

        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            self.roberta_model.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.roberta_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        raw_output = self.roberta_model(input_ids, attention_mask)
        # pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        # output = self.regressor(pooler)         # Shape is [batch_size, 1]
        return raw_output


def prediction_accuracy(data_with_predictions):
    data_dropna = data_with_predictions.dropna()
    data_dropna['gloss'] = data_dropna['gloss'].apply(lambda x: x[0])
    data_dropna['predicted_context'] = data_dropna['predicted_context'].apply(lambda x: x[0])
    return accuracy_score(data_dropna["gloss"], data_dropna["predicted_context"])


def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()

    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool


def calculate_wsd_accuracy(model, udpipe_model, eval_data, tokenizer):
    word_sense_detector = WordSenseDetector(
        pretrained_model=model,
        udpipe_model=udpipe_model,
        evaluation_dataset=eval_data,
        tokenizer=tokenizer,
        pooling_strategy=PoolingStrategy.mean_pooling,
        prediction_strategy=PredictionStrategy.all_examples_to_one_embedding
    )

    eval_data = word_sense_detector.run()
    return prediction_accuracy(eval_data)


def _calculate_cosine_scores(model, batch):
    anchor_ids = batch['anchor_ids'].to(device)
    anchor_mask = batch['anchor_mask'].to(device)
    pos_ids = batch['positive_ids'].to(device)
    pos_mask = batch['positive_mask'].to(device)

    a = model(anchor_ids, attention_mask=anchor_mask)[0]
    p = model(pos_ids, attention_mask=pos_mask)[0]

    a = mean_pool(a, anchor_mask)
    p = mean_pool(p, pos_mask)

    del anchor_ids, anchor_mask, pos_ids, pos_mask

    scores = torch.stack([cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a])
    return scores


def calculate_mnr_loss(model, batch):
    scores = _calculate_cosine_scores(model, batch)
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
    return cross_entropy_loss(scores * scale, labels)


def calculate_triplet_loss(model, batch):
    anchor_ids = batch['anchor_ids'].to(device)
    anchor_mask = batch['anchor_mask'].to(device)
    pos_ids = batch['positive_ids'].to(device)
    pos_mask = batch['positive_mask'].to(device)
    neg_ids = batch['negative_ids'].to(device)
    neg_mask = batch['negative_mask'].to(device)

    a = model(anchor_ids, attention_mask=anchor_mask)[0]
    p = model(pos_ids, attention_mask=pos_mask)[0]
    n = model(neg_ids, attention_mask=neg_mask)[0]

    a = mean_pool(a, anchor_mask)
    p = mean_pool(p, pos_mask)
    n = mean_pool(n, neg_mask)

    del anchor_ids, anchor_mask, pos_ids, pos_mask, neg_ids, neg_mask, batch

    return triplet_loss(a, p, n)


def tokenize_dataset(dataset, tokenizer, loss_name):
    dataset = dataset.map(
        lambda x: tokenizer(
            x["anchor"], max_length=128, padding='max_length',
            truncation=True
        ), batched=True
    )

    dataset = dataset.rename_column('input_ids', 'anchor_ids')
    dataset = dataset.rename_column('attention_mask', 'anchor_mask')

    dataset = dataset.map(
        lambda x: tokenizer(
            x['positive'], max_length=128, padding='max_length',
            truncation=True
        ), batched=True
    )

    dataset = dataset.rename_column('input_ids', 'positive_ids')
    dataset = dataset.rename_column('attention_mask', 'positive_mask')

    dataset = dataset.remove_columns(['anchor', 'positive'])

    if loss_name == "triplet":
        dataset = dataset.map(
            lambda x: tokenizer(
                x['negative'], max_length=128, padding='max_length',
                truncation=True
            ), batched=True
        )
        dataset = dataset.rename_column('input_ids', 'negative_ids')
        dataset = dataset.rename_column('attention_mask', 'negative_mask')
        dataset = dataset.remove_columns(['negative'])

    return dataset


def train(model, num_epochs, train_loader, eval_loader, udpipe_model, wsd_eval_data,
          tokenizer, run, loss_name, optim, scheduler, earle_stopping_rounds=30):

    os.mkdir('trained_models')

    batch_count = 0
    rounds_count = 0
    max_wsd_acc = 0

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)

        for batch in loop:

            # with torch.cuda.amp.autocast():
            if loss_name == "mnr":
                loss = calculate_mnr_loss(model, batch)
            elif loss_name == "triplet":
                loss = calculate_triplet_loss(model, batch)
            else:
                raise Exception("Undefined loss!")

            loss.backward()
            optim.step()
            optim.zero_grad()

            # # ⭐️⭐️ Gradient Accumulation
            # # Normalize the Gradients
            # scaler.scale(loss).backward()
            # loss = loss / NUM_ACCUMULATION_STEPS
            # if ((batch_count + 1) % NUM_ACCUMULATION_STEPS == 0) or (batch_count + 1 == len(train_loader)):
            #     scaler.step(optim)
            #     # optim.step()
            #     scaler.update()
            #     optim.zero_grad()

            scheduler.step()

            run["train/loss"].append(loss.item())

            report_gpu()
            if batch_count % 100 == 0:
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    loop = tqdm(eval_loader, leave=True)
                    for eval_batch in loop:
                        with torch.cuda.amp.autocast():
                            if loss_name == "mnr":
                                eval_loss += calculate_mnr_loss(model, eval_batch).item()
                            if loss_name == "triplet":
                                eval_loss += calculate_triplet_loss(model, eval_batch).item()
                        report_gpu()

                    print("Eval loss: " + str(round(eval_loss / len(eval_loader), 3)))

                    wsd_acc = calculate_wsd_accuracy(model, udpipe_model, wsd_eval_data, tokenizer)
                    report_gpu()
                    print("WSD accuracy: " + str(wsd_acc))

                    run["eval/wsd_acc"].append(wsd_acc)
                    run["eval/loss"].append(eval_loss / len(eval_loader))

                    if wsd_acc > max_wsd_acc:
                        max_wsd_acc = wsd_acc
                        rounds_count = 0
                        try:
                            if batch_count > 0:
                                model.save_pretrained(f"trained_models/model_{run.get_run_url().split('/')[-1][4:]}_{epoch}_{batch_count}", from_pt=True)
                        except:
                            print(f'model not saved epoch = {epoch}, batch = {batch_count}')

                    elif wsd_acc < max_wsd_acc:
                        rounds_count += 1

                    if rounds_count == earle_stopping_rounds:
                        print(f'Early stopping, model not improve WSD for {early_stopping}')
                        return

                model.train()
            batch_count += 1
            loop.set_description(f'Epoch {epoch}')

        try:
            model.save_pretrained(f"trained_models/model_{run.get_run_url().split('/')[-1][4:]}_{epoch}", from_pt=True)
        except:
            print(f'model not saved epoch = {epoch}, batch = {batch_count}')
        batch_count = 0


if __name__ == "__main__":
    run = neptune.init_run(
        project="vova.mudruy/WSD",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==",
    )

    batch_size = 128
    scale = 20.0
    learning_rate = 2e-6
    num_epochs = 10
    early_stopping = 50
    reinit_n_layers = 3
    loss_name = "triplet"
    NUM_ACCUMULATION_STEPS = 8

    udpipe_model = UDPipeModel("datasets_pre_defined/20180506.uk.mova-institute.udpipe")

    wsd_eval_data = pd.read_csv("datasets/wsd_loss_data_homonyms.csv")
    wsd_eval_data["examples"] = wsd_eval_data["examples"].apply(lambda x: literal_eval(x))
    wsd_eval_data["gloss"] = wsd_eval_data["gloss"].apply(lambda x: literal_eval(x))


    dataset = load_dataset('csv', data_files={'train': "datasets/wsd_lemma_homonyms_dataset_triplet_2m_filtered_30%_train_99.csv",
                                              'eval': "datasets/wsd_lemma_homonyms_dataset_triplet_2m_filtered_30%_eval_1.csv"})

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    dataset = tokenize_dataset(dataset, tokenizer, loss_name)

    dataset.set_format(type='torch', output_all_columns=True)

    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                                      output_hidden_states=True).to(device)
    
    model = nn.DataParallel(model)
    

    # model = ModelWithRandomizingSomeWeights(reinit_n_layers=reinit_n_layers).to(device)

    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(dataset["eval"], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optim, _ = roberta_base_AdamW_LLRD(model)

    # setup warmup for first ~10% of steps
    total_steps = int(len(dataset["train"]) / batch_size)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps - warmup_steps
    )

    run['epochs'] = num_epochs
    run['batch_size'] = batch_size
    run['scale'] = scale
    run["learning_rate"] = learning_rate
    run["early_stopping"] = early_stopping
    run["dataset/train"] = len(dataset["train"])
    run["dataset/eval"] = len(dataset["eval"])
    run["dataset/diff_threshold"] = 0.3
    # run["NUM_ACCUMULATION_STEPS"] = NUM_ACCUMULATION_STEPS

    train(model, num_epochs, train_loader, eval_loader, udpipe_model, wsd_eval_data, tokenizer, run, loss_name, optim,
          scheduler, earle_stopping_rounds=early_stopping)
    run.stop()
