import multiprocessing as mp
import string
import tqdm
import smart_open
import json
from langdetect import detect
from  src.udpipe_model import UDPipeModel
from src.config import PATH_TO_SOURCE_DATASET, PATH_TO_SOURCE_UDPIPE, PATH_TO_SAVE_GATHERED_DATASET, \
    PATH_TO_LEMMAS_OF_INTEREST, NUMBER_OF_EXAMPLES_TO_GATHER

udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)


class CollectUberTextTriplets:
    def __init__(self, path_to_ubertext, path_to_save_gathered_dataset, path_to_lemmas_of_interest,
                 number_of_examples_to_gather):
        self.path_to_ubertext = path_to_ubertext
        self.path_to_save_gathered_dataset = path_to_save_gathered_dataset
        self.path_to_lemmas_of_interest = path_to_lemmas_of_interest
        self.number_of_examples_to_gather = number_of_examples_to_gather

        with open(self.path_to_lemmas_of_interest) as f:
            unique_lemmas = f.readlines()
        self.lemmas_of_interest = set([i.replace("\n", "") for i in unique_lemmas])

    def _normalize_text(self, line):
        global udpipe_model
        tokens = udpipe_model.tokenize(line)

        for tok_sent in tokens:
            udpipe_model.tag(tok_sent)
        return {w.lemma for w in tok_sent.words[1:]}

    def _save_raw_examples_to_json(self, data):
        with open(self.path_to_save_gathered_dataset, 'w') as f:
            json.dump(data, f)

    def _process_ubertext_line(self, line):
        line = line.replace("\n", "").replace("\xa0", " ").strip()

        line_ = line.translate(str.maketrans('', '', string.punctuation))  # remove punctuation from a processing line
        line_ = ' '.join(line_.split())

        lint_split = line_.split(" ")
        if len(lint_split) <= 7 or len(lint_split) >= 16:
            return None

        if line.count("*") >= 4:
            return None

        if line.count("—") >= 5:
            return None

        if sum(c.isdigit() for c in line) >= 10:
            return None

        try:
            if detect(line) != "uk":
                return None
        except Exception as e:
            return None

        nomalize_line = self._normalize_text(line)
        intersection = self.lemmas_of_interest.intersection(nomalize_line)

        if len(intersection) > 0:
            return (intersection, line)

        return None

    def _collect_raw_lemma_examples_dataset(self):
        pool = mp.Pool(processes=mp.cpu_count())

        lemma_examples = {}
        total_number_of_sentences = 0

        with smart_open.open(self.path_to_ubertext, encoding='utf-8') as f:

            for lemma_sentence in tqdm.tqdm(pool.imap_unordered(self._process_ubertext_line, f)):
                if lemma_sentence is not None:
                    total_number_of_sentences += 1

                    for lemma in lemma_sentence[0]:
                        if lemma in lemma_examples.keys():
                            lemma_examples[lemma].append(lemma_sentence[1])
                        else:
                            lemma_examples[lemma] = [lemma_sentence[1]]

                if total_number_of_sentences == self.number_of_examples_to_gather:
                    break

        pool.close()
        pool.join()

        return lemma_examples

    def collect_triplets(self):
        raw_lemma_examples = self._collect_raw_lemma_examples_dataset()
        self._save_raw_examples_to_json(raw_lemma_examples)


if __name__ == "__main__":
    collector = CollectUberTextTriplets(PATH_TO_SOURCE_DATASET, PATH_TO_SAVE_GATHERED_DATASET,
                                        PATH_TO_LEMMAS_OF_INTEREST, NUMBER_OF_EXAMPLES_TO_GATHER)
    collector.collect_triplets()