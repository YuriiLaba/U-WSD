import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel

from collections import Counter
from scipy.spatial import distance

from src.poolings import PoolingStrategy
from src.utils_model import get_hidden_states


class WordSenseDetector:

    def __init__(self, pretrained_model, udpipe_model, evaluation_dataset,
                 prediction_strategy="all_examples_to_one_embedding", pooling_strategy="mean_pooling", **kwargs):
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
        self.missing_target_word_in_sentence = 0
        # TODO create a polling selection

    def tokenize_text(self, input_text):
        return self.tokenizer(input_text, return_tensors='pt').to(self.device)

    def find_target_word_in_sentence(self, input_text, target_word):
        """
        TODO: This function wil only find one target word
        """

        # TODO fix misunderstandin issue with similarity of tokenize and tokenize_text
        tokenized = self.udpipe_model.tokenize(input_text)

        for tok_sent in tokenized:

            self.udpipe_model.tag(tok_sent)

            for word_index, w in enumerate(tok_sent.words[1:]):  # under 0 index is root
                # print(w.lemma)

                if w.lemma.lower() == target_word.lower():
                    # TODO check if this better return tok_sent.words[word_index+1].form
                    return [w.form for w in tok_sent.words[1:]][word_index]

        self.missing_target_word_in_sentence += 1
        return None

    def find_target_word_in_tokenized_text(self, tokenized_input_text, word):
        # TODO check if better list(set(tokenized_input_text.word_ids()))
        unique_w_ids = list(Counter(tokenized_input_text.word_ids()).keys())
        unique_w_ids.remove(None)
        tokens = ["" for _ in range(len(unique_w_ids))]

        word_indexes = []
        prev_word_id = None

        target_words_with_indexes = []

        for index, (input_id, word_id) in enumerate(zip(tokenized_input_text["input_ids"][0],
                                                        tokenized_input_text.word_ids())):
            token = self.tokenizer.decode([input_id]).strip()

            if prev_word_id == word_id:
                word_indexes.append(index)

            else:
                word_indexes = []
                word_indexes.append(index)

            prev_word_id = word_id

            if word_id is None:
                continue

            if token.startswith("##"):
                token = token.replace("##", "")

            tokens[word_id] += token
            if tokens[word_id].replace("▁", "").lower() == word.lower():
                target_words_with_indexes.append((word, word_indexes.copy()))
        # print(tokens)

        return target_words_with_indexes

    def get_model_output(self, tokenized_input_text):
        with torch.no_grad():
            model_output = self.model(**tokenized_input_text)
        return model_output

    def run_inference(self, text):
        tokenized_text = self.tokenize_text(text)
        model_output = self.get_model_output(tokenized_text)
        return tokenized_text, get_hidden_states(model_output)

    def get_target_word_embedding(self, target_word, sentence_example):
        # TODO: remove it
        ACUTE = chr(0x301)
        GRAVE = chr(0x300)
        target_word_fixed = target_word.replace(GRAVE, "").replace(ACUTE, "")

        # TODO rename "word" for better understanding
        word = self.find_target_word_in_sentence(sentence_example, target_word_fixed)
        if word is None:
            # print("Can't find target word in sentence")
            return None

        tokenized_input_text, hidden_states = self.run_inference(sentence_example)
        target_word_indexes = self.find_target_word_in_tokenized_text(tokenized_input_text, word)

        # TODO if len(target_word_indexes) != 1:
        if len(target_word_indexes) == 0:
            # print("Cant find target word in tokens")
            return None
        # TODO we drop all word if there are more that 2 target word occurence in example - its bad
        if len(target_word_indexes) > 1:
            # print("Skip for now")
            return None

        target_word_indexes = target_word_indexes[0][1]  # TODO: explain
        if self.pooling_strategy == "mean_pooling":
            return PoolingStrategy.mean_pooling(hidden_states[target_word_indexes[0]:target_word_indexes[-1] + 1])

        elif self.pooling_strategy == "max_pooling":
            return PoolingStrategy.max_pooling(hidden_states[target_word_indexes[0]:target_word_indexes[-1] + 1])

        elif self.pooling_strategy == "mean_max_pooling":
            return PoolingStrategy.mean_max_pooling(hidden_states[target_word_indexes[0]:target_word_indexes[-1] + 1])

        elif self.pooling_strategy == "concatenate_pooling":
            return PoolingStrategy.concatenate_pooling(hidden_states[target_word_indexes[0]:target_word_indexes[-1] + 1])

        elif self.pooling_strategy == "last_four_sum_pooling":
            return PoolingStrategy.last_four_sum_pooling(hidden_states[target_word_indexes[0]:target_word_indexes[-1] + 1])

        elif self.pooling_strategy == "last_two_sum_pooling":
            return PoolingStrategy.last_two_sum_pooling(hidden_states[target_word_indexes[0]:target_word_indexes[-1] + 1])

    def get_context_embedding(self, context):
        _, hidden_states_context = self.run_inference(context)
        if self.pooling_strategy == "mean_pooling":
            return PoolingStrategy.mean_pooling(hidden_states_context)

        elif self.pooling_strategy == "max_pooling":
            return PoolingStrategy.max_pooling(hidden_states_context)

        elif self.pooling_strategy == "mean_max_pooling":
            return PoolingStrategy.mean_max_pooling(hidden_states_context)

        elif self.pooling_strategy == "concatenate_pooling":
            return PoolingStrategy.concatenate_pooling(hidden_states_context)

        elif self.pooling_strategy == "last_four_sum_pooling":
            return PoolingStrategy.last_four_sum_pooling(hidden_states_context)

        elif self.pooling_strategy == "last_two_sum_pooling":
            return PoolingStrategy.last_two_sum_pooling(hidden_states_context)

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
                target_word_embedding = self.get_target_word_embedding(lemma, example)
                if target_word_embedding is None:
                    return None

                combined_embedding.append(target_word_embedding)

            combined_embedding = np.asarray(combined_embedding)
            combined_embedding = np.mean(combined_embedding, axis=0)

            for context in contexts:
                max_sub_sim = -1

                for sub_context in context:
                    sub_context_embedding = self.get_context_embedding(sub_context)
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

            for context in contexts:
                context_embedding = self.get_context_embedding(context)

                for example in examples:
                    target_word_embedding = self.get_target_word_embedding(lemma, example)
                    if target_word_embedding is None:
                        return None

                    similarity = 1 - distance.cosine(target_word_embedding, context_embedding)

                    if similarity > max_sim:
                        max_sim = similarity
                        correct_context = context

            return correct_context

    def run(self):
        # TODO groupby data by lemma and create list of all contexts (this will help to remove data from init and move to run)
        # TODO rewrite apply to only 2 columns and use params (lemma-senses dict) .apply(lambda x: self.predict_word_sense(x, param), axis=1)
        self.evaluation_dataset["predicted_context"] = self.evaluation_dataset.apply(self.predict_word_sense, axis=1)
        return self.evaluation_dataset
