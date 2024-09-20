import json
import pandas as pd
import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

# TODO: rename this variable
from services.config import PATH_TO_SAVE_GATHERED_DATASET, EMBEDDER_MODEL, TRIPLET_PROCESSOR_BATCH_SIZE, \
    PATH_TO_SAVE_TRIPLETS

with open(PATH_TO_SAVE_GATHERED_DATASET, 'r') as f:
    lemma_examples_dataset = json.load(f)

embedder = SentenceTransformer(EMBEDDER_MODEL)


def fill_diagonal(scores, value, shift):
    for i in range(len(scores)):
        if i+shift > scores.shape[1]-1:
            break
        scores[i, i+shift] = value
    return scores


triplets = pd.DataFrame(columns=['lemma', 'anchor', 'positive', 'negative', 'positive_score', 'negative_score'])

for lemma in tqdm.tqdm(lemma_examples_dataset.keys()):

    if len(lemma_examples_dataset[lemma]) < 3:
        continue

    sentences = list(set(lemma_examples_dataset[lemma]))

    start = 0
    end = TRIPLET_PROCESSOR_BATCH_SIZE

    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

    while start < len(sentences):
        cosine_scores = util.cos_sim(sentence_embeddings[start:end], sentence_embeddings)

        cosine_scores = fill_diagonal(cosine_scores, -1, start)
        positive_scores, positive_indexes = torch.max(cosine_scores, axis=1)
        positive = [sentences[i] for i in positive_indexes]

        cosine_scores = fill_diagonal(cosine_scores, 1, start)
        negative_scores, negative_indexes = torch.min(cosine_scores, axis=1)
        negative = [sentences[i] for i in negative_indexes]

        lemma_df = pd.DataFrame({
            "lemma": lemma,
            "anchor": sentences[start:end],
            "positive": positive,
            "negative": negative,
            "positive_score": positive_scores.cpu(),
            "negative_score": negative_scores.cpu()
        })
        triplets = triplets.append(lemma_df)

        start += TRIPLET_PROCESSOR_BATCH_SIZE
        end += TRIPLET_PROCESSOR_BATCH_SIZE

triplets.to_csv(PATH_TO_SAVE_TRIPLETS, index=False)
