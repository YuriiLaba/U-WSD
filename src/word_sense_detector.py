import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

tqdm.pandas()


class WordSenseDetector:

    def __init__(self, pretrained_model, udpipe_model, evaluation_dataset, pooling_strategy,
                 prediction_strategy, **kwargs):
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

        return self.prediction_strategy(lemma, examples, contexts, self.model, self.tokenizer, self.udpipe_model,
                                        self.pooling_strategy, self.device)

    def run(self):
        # TODO groupby data by lemma and create list of all contexts (this will help to remove data from init and move to run)
        # TODO rewrite apply to only 2 columns and use params (lemma-senses dict) .apply(lambda x: self.predict_word_sense(x, param), axis=1)
        self.evaluation_dataset["predicted_context"] = self.evaluation_dataset.progress_apply(self.predict_word_sense, axis=1)
        return self.evaluation_dataset
