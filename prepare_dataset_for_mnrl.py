import torch
import pandas as pd
from smart_open import open
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('youscan/ukr-roberta-base')

sentences_dataset = {
    "anchor": [],
    "positive": []
}


count = 0

for line in open("all_uniq_filtered_shuffled.txt.bz2"):
    if count == 2000000:
        break

    sentences_dataset["anchor"].append(line)
    count += 1


embeddings = model.encode(sentences_dataset["anchor"], convert_to_tensor=True)

for count, i in enumerate(range(len(embeddings))):

    if count % 10000 == 0:
        print(count)

    cosine_scores = util.cos_sim(embeddings[i], embeddings)

    cosine_scores[0][i] = -1
    most_similar_idx = torch.argmax(cosine_scores[0]).cpu().numpy()

    cosine_score = cosine_scores[0][most_similar_idx].cpu().numpy()

    sentences_dataset["positive"].append(sentences_dataset["anchor"][most_similar_idx])
    sentences_dataset["cos_distance"].append(cosine_score)

sentences_dataset_pd = pd.DataFrame.from_dict(sentences_dataset)

sentences_dataset_pd.to_csv("2m_cosine_distances_max.csv", index=False)