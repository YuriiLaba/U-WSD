from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from transformers.optimization import get_linear_schedule_with_warmup
import neptune.new as neptune
import torch
from tqdm.auto import tqdm
import pandas as pd
from ast import literal_eval

import torch.nn as nn
import torch.nn.functional as F

from src.word_sense_detector import WordSenseDetector
from src.udpipe_model import UDPipeModel
from sklearn.metrics import accuracy_score


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cos_sim = torch.nn.CosineSimilarity().to(device)
cross_entropy_loss = torch.nn.CrossEntropyLoss().to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)


def prediction_accuracy(data_with_predictions):
    data_dropna = data_with_predictions.dropna()

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
        tokenizer=tokenizer)

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
          tokenizer, run, loss_name, earle_stopping_rounds=30):
    batch_count = 0
    rounds_count = 0
    max_wsd_acc = 0

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)

        for batch in loop:

            optim.zero_grad()

            if loss_name == "mnr":
                loss = calculate_mnr_loss(model, batch)
            elif loss_name == "triplet":
                loss = calculate_triplet_loss(model, batch)
            else:
                raise Exception("Undefined loss!")

            loss.backward()
            optim.step()
            scheduler.step()
            run["train/loss"].append(loss.item())

            if batch_count % 100 == 0:
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        if loss_name == "mnr":
                            eval_loss = calculate_mnr_loss(model, eval_batch)
                        if loss_name == "triplet":
                            eval_loss = calculate_triplet_loss(model, eval_batch)

                    print("Eval loss: " + str(round(eval_loss / len(eval_loader), 3)))

                    wsd_acc = calculate_wsd_accuracy(model, udpipe_model, wsd_eval_data, tokenizer)
                    print("WSD accuracy: " + str(wsd_acc))

                    run["eval/wsd_acc"].append(wsd_acc)
                    run["eval/loss"].append(eval_loss / len(eval_loader))

                    if wsd_acc > max_wsd_acc:
                        max_wsd_acc = wsd_acc
                        rounds_count = 0
                        model.save_pretrained("pytorch_model", from_pt=True)
                    elif wsd_acc < max_wsd_acc:
                        rounds_count += 1

                    if rounds_count == earle_stopping_rounds:
                        print(f'Early stopping, model not improve WSD for {early_stopping}')
                        return

                model.train()
            batch_count += 1
            loop.set_description(f'Epoch {epoch}')

        model.save_pretrained(f'pytorch_model {epoch}', from_pt=True)
        batch_count = 0


if __name__ == "__main__":
    run = neptune.init_run(
        project="vova.mudruy/WSD",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==",
    )

    batch_size = 32
    scale = 20.0
    learning_rate = 2e-5
    num_epochs = 10
    early_stopping = 70
    loss_name = "triplet"

    udpipe_model = UDPipeModel("20180506.uk.mova-institute.udpipe")

    wsd_eval_data = pd.read_csv("wsd_loss_data_homonyms.csv")
    wsd_eval_data["examples"] = wsd_eval_data["examples"].apply(lambda x: literal_eval(x))

    dataset = load_dataset('csv', data_files={'train': "wsd_lemma_homonyms_dataset_triplet_train_95.csv",
                                              'eval': "wsd_lemma_homonyms_dataset_triplet_eval_5.csv"})

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    dataset = tokenize_dataset(dataset, tokenizer, loss_name)

    dataset.set_format(type='torch', output_all_columns=True)

    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                                      output_hidden_states=True).to(device)

    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset["eval"], batch_size=batch_size, shuffle=True)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    train(model, num_epochs, train_loader, eval_loader, udpipe_model, wsd_eval_data, tokenizer, run, loss_name,
          earle_stopping_rounds=early_stopping)
    run.stop()
