from sentence_transformers import SentenceTransformer, util
from sentence_transformers import InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader
import pandas as pd

dataset = pd.read_csv("")
fine_tune_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

train_examples = []

for anchor, positive, negative in zip(dataset["anchor"], dataset["positive"], dataset["negative"]):
    train_examples.append(InputExample(texts=[anchor, positive, negative]))

train_dataset = SentencesDataset(train_examples, fine_tune_model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
train_loss = losses.TripletLoss(model=fine_tune_model)
fine_tune_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)
fine_tune_model.save("paraphrase_multilingual_mpnet_base_v2_lemma_mistaken")