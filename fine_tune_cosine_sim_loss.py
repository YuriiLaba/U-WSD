from sentence_transformers import SentenceTransformer, util
from sentence_transformers import InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader

import pandas as pd

cosine_distances_150k = pd.read_csv("150k_cosine_distances.csv")

fine_tune_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

train_examples = []

for anchor, positive, cos_distance in zip(cosine_distances_150k["anchor"],
                                          cosine_distances_150k["positive"],
                                          cosine_distances_150k["cos_distance"]):

    train_examples.append(InputExample(texts=[anchor, positive], label=cos_distance))

train_dataset = SentencesDataset(train_examples, fine_tune_model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model=fine_tune_model)
fine_tune_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)

fine_tune_model.save("paraphrase_multilingual_mpnet_base_v2_150k_cosine")