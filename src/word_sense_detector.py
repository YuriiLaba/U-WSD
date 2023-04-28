import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

from transformers import AutoTokenizer, AutoModel

from src.utils_embedding_calculation import get_target_word_embedding, get_context_embedding

tqdm.pandas()


class WordSenseDetector:

    def __init__(self, pretrained_model, udpipe_model, evaluation_dataset, pooling_strategy,
                 prediction_strategy="all_examples_to_one_embedding", **kwargs):
        # TODO create doc-string especially for describing prediction_strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(pretrained_model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.model = AutoModel.from_pretrained(pretrained_model, output_hidden_states=True).to(self.device)
        else:
            self.tokenizer = kwargs["tokenizer"]
            self.model = pretrained_model.to(self.device)
        self.udpipe_model = udpipe_model
        self.evaluation_dataset = evaluation_dataset
        self.prediction_strategy = prediction_strategy
        self.pooling_strategy = pooling_strategy
        # TODO create a WSD_logger and move there missing_target_word_in_sentence

    def predict_word_sense(self, row):

        lemma = row["lemma"]
        examples = row["examples"]
        contexts = self.evaluation_dataset[self.evaluation_dataset["lemma"] == lemma]["gloss"].tolist()

        # TODO create as 2 separate methods
        if self.prediction_strategy == "all_examples_to_one_embedding":

            # TODO: check whether it's more efficient to create numpy here
            combined_embedding = []

            max_sim = -1
            correct_context = None

            for example in examples:
                target_word_embedding = get_target_word_embedding(self.model, self.tokenizer, self.udpipe_model,
                                                                  self.pooling_strategy, lemma, example, self.device)
                if target_word_embedding is not None:
                    combined_embedding.append(target_word_embedding)

            if len(combined_embedding) == 0:
                return None

            combined_embedding = np.asarray(combined_embedding)
            combined_embedding = np.mean(combined_embedding, axis=0)

            for context in contexts:
                max_sub_sim = -1

                for sub_context in context:
                    sub_context_embedding = get_context_embedding(self.model, self.tokenizer, self.pooling_strategy,
                                                                  sub_context, self.device)
                    sub_similarity = 1 - distance.cosine(combined_embedding, sub_context_embedding)

                    if sub_similarity > max_sub_sim:
                        max_sub_sim = sub_similarity

                if max_sub_sim > max_sim:
                    max_sim = max_sub_sim
                    correct_context = context
            return correct_context

        if self.prediction_strategy == "max_sim_across_all_examples":

            max_sim = -1
            correct_context = None
            target_word_embeddings = []

            for example in examples:
                target_word_embedding = get_target_word_embedding(self.model, self.tokenizer, self.udpipe_model,
                                                                  self.pooling_strategy, lemma, example, self.device)
                if target_word_embedding is not None:
                    target_word_embeddings.append(target_word_embedding)

            for context in contexts:
                max_sub_sim = -1

                for sub_context in context:
                    sub_context_embedding = get_context_embedding(self.model, self.tokenizer, self.pooling_strategy,
                                                                  sub_context, self.device)

                    for embedding in target_word_embeddings:
                        sub_similarity = 1 - distance.cosine(embedding, sub_context_embedding)
                        if sub_similarity > max_sub_sim:
                            max_sub_sim = sub_similarity

                if max_sub_sim > max_sim:
                    max_sim = max_sub_sim
                    correct_context = context

            return correct_context

    def run(self):
        # TODO groupby data by lemma and create list of all contexts (this will help to remove data from init and move to run)
        # TODO rewrite apply to only 2 columns and use params (lemma-senses dict) .apply(lambda x: self.predict_word_sense(x, param), axis=1)
        self.evaluation_dataset["predicted_context"] = self.evaluation_dataset.progress_apply(self.predict_word_sense, axis=1)
        return self.evaluation_dataset
