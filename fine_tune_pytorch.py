from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from transformers.optimization import get_linear_schedule_with_warmup
import neptune.new as neptune
import torch
from tqdm.auto import tqdm

run = neptune.init_run(
    project="vova.mudruy/WSD",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==",
)


def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()

    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool


def calculate_cosine_scores(model, batch):
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


dataset = load_dataset('csv', data_files={'train': "500k_cosine_distances_train_gt_70_lt_90.csv",
                                          'eval': "500k_cosine_distances_eval_gt_70_lt_90.csv"})

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

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
dataset.set_format(type='torch', output_all_columns=True)

batch_size = 32

train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size)
eval_loader = torch.utils.data.DataLoader(dataset["eval"], batch_size=batch_size)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f'moved to {device}')

cos_sim = torch.nn.CosineSimilarity()
loss_func = torch.nn.CrossEntropyLoss()
scale = 20.0

cos_sim.to(device)
loss_func.to(device)

# initialize Adam optimizer
optim = torch.optim.Adam(model.parameters(), lr=2e-5)

# setup warmup for first ~10% of steps
total_steps = int(len(dataset["train"]) / batch_size)
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optim, num_warmup_steps=warmup_steps,
    num_training_steps=total_steps - warmup_steps
)

num_epochs = 10
run['epochs'] = num_epochs
run['batch_size'] = batch_size

batch_count = 0
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)

    train_loss = 0

    for batch in loop:

        optim.zero_grad()

        scores = calculate_cosine_scores(model, batch)

        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)

        loss = loss_func(scores * scale, labels)

        loss.backward()
        optim.step()
        scheduler.step()

        run["train/loss"].append(loss.item())

        if batch_count % 100 == 0:
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    scores = calculate_cosine_scores(model, eval_batch)
                    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
                    eval_loss += loss_func(scores * scale, labels).item()
                print("Eval loss: " + str(round(eval_loss / len(eval_loader), 3)))

                del labels, scores

                run["eval/loss"].append(eval_loss / len(eval_loader))
            model.train()
        batch_count += 1
        loop.set_description(f'Epoch {epoch}')

    batch_count = 0
    model.save(f"pytorch_model_{epoch}")

run.stop()
