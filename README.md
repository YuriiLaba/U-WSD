# Contextual embeddings for Ukrainian: A large language model approach to word sense disambiguation


[![Download model 🤗 ](https://img.shields.io/badge/Download%20Model-%F0%9F%A4%97%20Model-yellow)](https://huggingface.co/lang-uk/ukr-paraphrase-multilingual-mpnet-base)

This repository is an official implementation of paper
[Contextual Embeddings for Ukrainian: A Large Language Model Approach to Word Sense Disambiguation](https://aclanthology.org/2023.unlp-1.2/) (Laba et al., UNLP 2023 at EACL).

## WSD task overview
Word Sense Disambiguation (WSD) task involves identifying a polysemic word’s correct meaning in a given context.

## Solution overview
In our approach to the WSD task, we fine-tuned the [S-BERT model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) for the Ukrainian language. The model was fine-tuned using the unsupervised triplet dataset built on [UberText2.0](https://lang.org.ua/en/ubertext/). We used TripletMarginLoss during fine-tuning to maximize the separation between correct and incorrect senses. 

The model was validated on a homonyms dataset based on the [СЛОВНИК УКРАЇНСЬКОЇ МОВИ](https://sum20ua.com). To the best of our knowledge, this is the first dataset to validate the WSD task in Ukrainian.

The model can be utilized to generate high-quality embeddings for the Ukrainian language. It is available on [Hugging Face]((https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)).


## Metrics
| Model                          | Overall Acc. | Noun Acc. | Verb Acc. | Adj. Acc. | Adv. Acc. |
|--------------------------------|--------------|-----------|-----------|-----------|-----------|
| Babelfy Baseline               | 0.526        | -         | -         | -         | -         |
| PMMBv2                         | 0.735        | 0.767     | 0.668     | 0.752     | 0.593     |
| PMMBv2 Tuned on ∼190K Triplets | 0.770        | 0.819     | 0.685     | 0.743     | 0.562     |
| PMMBv2 Tuned on ∼1.2M Triplets | 0.778        | 0.825     | 0.698     | 0.761     | 0.531     |
| PMMBv2 Tuned on ∼1.2M Triplets with Filtering | 0.779        | 0.824     | 0.693     | 0.759     | 0.607     |
## Datasets
### WSD Dataset
### Train Dataset
## How to reproduce the results
### Step 1: Preparation
### Step 2: Train
### Step 3: Evaluation

## Citation


TODO: publish paper on papers with code

TODO: publish WSD eval dataset and link it