import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel

from collections import Counter
from scipy.spatial import distance


class WordSenseDetector:

    def __init__(self, pretrained_model, udpipe_model, evaluation_dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(pretrained_model, output_hidden_states=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.udpipe_model = udpipe_model
        self.evaluation_dataset = evaluation_dataset
        self.missing_target_word_in_sentence = 0

    def tokenize_text(self, input_text):
        return self.tokenizer(input_text, return_tensors='pt').to(self.device)

    def find_target_word_in_sentence(self, input_text, target_word):
        """
        TODO: This function wil only find one target word
        """

        tokenized = self.udpipe_model.tokenize(input_text)

        for tok_sent in tokenized:

            self.udpipe_model.tag(tok_sent)

            for word_index, w in enumerate(tok_sent.words[1:]):  # under 0 index is root
                # print(w.lemma)

                if w.lemma.lower() == target_word.lower():
                    return [w.form for w in tok_sent.words[1:]][word_index]

        self.missing_target_word_in_sentence += 1
        return None

    def find_target_word_in_tokenized_text(self, tokenized_input_text, word):
        unique_w_ids = list(Counter(tokenized_input_text.word_ids()).keys())
        unique_w_ids.remove(None)
        tokens = ["" for i in range(len(unique_w_ids))]

        word_indexes = []
        prev_word_id = None

        target_words_with_indexes = []

        # for index, (token, word_id) in enumerate(zip(self.tokenizer.convert_ids_to_tokens(tokenized_input_text.input_ids[0]), tokenized_input_text.word_ids())):

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

    def run_inference(self, tokenized_input_text):
        with torch.no_grad():
            model_output = self.model(**tokenized_input_text)
        return model_output

    def get_hidden_states(self, model_output):
        hidden_states = model_output["hidden_states"]

        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = torch.squeeze(hidden_states, dim=1)
        hidden_states = hidden_states.permute(1,0,2)

        return hidden_states

    def get_last_hidden_state(self, hidden_states):
        return hidden_states[:, -1]

    def mean_pooling(self, hidden_states):
        last_hidden_state = self.get_last_hidden_state(hidden_states)
        return torch.mean(last_hidden_state, dim=0).cpu().detach().numpy()

    def get_embedding_of_the_target_word(self, hidden_states, target_word_indexes):
        return self.mean_pooling(hidden_states[target_word_indexes[0]:target_word_indexes[-1]+1])

    def predict_word_sense(self, row):
        ACUTE = chr(0x301)
        GRAVE = chr(0x300)

        lemma = row["lemma"]
        example = row["example"]

        lemma_fixed = lemma.replace(GRAVE, "").replace(ACUTE, "")

        word = self.find_target_word_in_sentence(example, lemma_fixed)
        if word == None:
            # print("Can't find target word in sentence")
            return None

        tokenized_input_text = self.tokenize_text(example)
        word_in_tokens = self.find_target_word_in_tokenized_text(tokenized_input_text, word)

        if len(word_in_tokens) == 0:
            # print("Cant find target word in tokens")
            return None
        if len(word_in_tokens) > 1:
            # print("Skip for now")
            return None

        model_output = self.run_inference(tokenized_input_text)
        hidden_states = self.get_hidden_states(model_output)

        contexts = self.evaluation_dataset[self.evaluation_dataset["lemma"] == lemma]["gloss"].tolist()

        max_sim = -1
        correct_context = None

        for word_info in word_in_tokens:
            word_ = word_info[0]
            word_indexes_ = word_info[1]

            word_embedding = self.get_embedding_of_the_target_word(hidden_states, word_indexes_)

            for context in contexts:

                tokenized_input_text_context = self.tokenize_text(context)

                model_output_context = self.run_inference(tokenized_input_text_context)
                hidden_states_context = self.get_hidden_states(model_output_context)

                context_embedding = self.mean_pooling(hidden_states_context)

                similarity = 1 - distance.cosine(word_embedding, context_embedding)
                if similarity > max_sim:
                    max_sim = similarity
                    correct_context = context

        return correct_context

    def run(self):
        self.evaluation_dataset["predicted_context"] = self.evaluation_dataset.apply(self.predict_word_sense, axis=1)
        return self.evaluation_dataset
