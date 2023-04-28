import numpy as np
from src.utils_embedding_calculation import get_target_word_embedding, get_context_embedding
from scipy.spatial import distance


class PredictionStrategy:
    @staticmethod
    def all_examples_to_one_embedding(lemma, examples, contexts, model, tokenizer, udpipe_model, pooling_strategy, device):

        # TODO: check whether it's more efficient to create numpy here
        combined_embedding = []

        max_sim = -1
        correct_context = None

        for example in examples:
            target_word_embedding = get_target_word_embedding(model, tokenizer, udpipe_model, pooling_strategy, lemma,
                                                              example, device)
            if target_word_embedding is not None:
                combined_embedding.append(target_word_embedding)

        if len(combined_embedding) == 0:
            return None

        combined_embedding = np.asarray(combined_embedding)
        combined_embedding = np.mean(combined_embedding, axis=0)

        for context in contexts:
            max_sub_sim = -1

            for sub_context in context:
                sub_context_embedding = get_context_embedding(model, tokenizer, pooling_strategy, sub_context, device)
                sub_similarity = 1 - distance.cosine(combined_embedding, sub_context_embedding)

                if sub_similarity > max_sub_sim:
                    max_sub_sim = sub_similarity

            if max_sub_sim > max_sim:
                max_sim = max_sub_sim
                correct_context = context
        return correct_context

    @staticmethod
    def max_sim_across_all_examples(lemma, examples, contexts, model, tokenizer, udpipe_model, pooling_strategy, device):

        max_sim = -1
        correct_context = None
        target_word_embeddings = []

        for example in examples:
            target_word_embedding = get_target_word_embedding(model, tokenizer, udpipe_model, pooling_strategy, lemma,
                                                              example, device)
            if target_word_embedding is not None:
                target_word_embeddings.append(target_word_embedding)

        for context in contexts:
            max_sub_sim = -1

            for sub_context in context:
                sub_context_embedding = get_context_embedding(model, tokenizer, pooling_strategy, sub_context, device)

                for embedding in target_word_embeddings:
                    sub_similarity = 1 - distance.cosine(embedding, sub_context_embedding)
                    if sub_similarity > max_sub_sim:
                        max_sub_sim = sub_similarity

            if max_sub_sim > max_sim:
                max_sim = max_sub_sim
                correct_context = context

        return correct_context
