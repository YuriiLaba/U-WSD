from sentence_transformers import SentenceTransformer, util
from sentence_transformers import InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader
import neptune.new as neptune

CUDA_VISIBLE_DEVICES=1,2

import pandas as pd
run = neptune.init_run(
    project="vova.mudruy/WSD",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTg0NWQxYy0zNTVkLTQwZDktODJhZC00ZjgxNGNhODE2OTIifQ==",
)
cosine_distances_2m = pd.read_csv("2m_cosine_distances_max.csv")

cosine_distances_2m = cosine_distances_2m[cosine_distances_2m["cos_distance"] > 0.85]

fine_tune_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

train_examples = []

for anchor, positive in zip(cosine_distances_2m["anchor"], cosine_distances_2m["positive"]):

    train_examples.append(InputExample(texts=[anchor, positive]))


run['epochs'] = 5
run['batch_size'] = 32
train_dataset = SentencesDataset(train_examples, fine_tune_model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model=fine_tune_model)
fine_tune_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5)

fine_tune_model.save("paraphrase_multilingual_mpnet_base_v2_2m_cosine_mult_neg_rank_loss")
run.stop()