# Contextual embeddings for Ukrainian: A large language model approach to word sense disambiguation


[![Download model 🤗 ](https://img.shields.io/badge/Download%20Model-%F0%9F%A4%97%20Model-yellow)](https://huggingface.co/lang-uk/ukr-paraphrase-multilingual-mpnet-base)

This repository is an official implementation of paper
[Contextual Embeddings for Ukrainian: A Large Language Model Approach to Word Sense Disambiguation](https://aclanthology.org/2023.unlp-1.2/) (Laba et al., UNLP 2023 at EACL).

## WSD task overview
Word Sense Disambiguation (WSD) task involves identifying a polysemic word’s correct meaning in a given context.

## Solution overview

In our approach to the Word Sense Disambiguation (WSD) task, we fine-tuned the **ConEFU** model for the Ukrainian language. **ConEFU** is based on the [S-BERT model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) and was fine-tuned using an unsupervised triplet dataset built on [UberText2.0](https://lang.org.ua/en/ubertext/). We employed TripletMarginLoss during fine-tuning to maximize the separation between correct and incorrect senses of homonyms.

The model was validated on a homonyms dataset based on the [СЛОВНИК УКРАЇНСЬКОЇ МОВИ](https://sum20ua.com). To the best of our knowledge, this is the first dataset used to validate the WSD task in Ukrainian.

**ConEFU** can be utilized to generate high-quality embeddings for the Ukrainian language. It is available on [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2).



## Metrics
| Model                          | Overall Acc. | Noun Acc. | Verb Acc. | Adj. Acc. | Adv. Acc. |
|--------------------------------|--------------|-----------|-----------|-----------|-----------|
| Babelfy Baseline               | 0.526        | -         | -         | -         | -         |
| PMMBv2                         | 0.735        | 0.767     | 0.668     | 0.752     | 0.593     |
| ConEFU ∼190K Triplets | 0.770        | 0.819     | 0.685     | 0.743     | 0.562     |
| ConEFU ∼1.2M Triplets | 0.778        | **0.825**     | **0.698**     | **0.761**     | 0.531     |
| ConEFU ∼1.2M Triplets with Filtering | **0.779**        | 0.824     | 0.693     | 0.759     | **0.607**     |
## Datasets

### The WSD evaluation dataset

- **Lemma**: The base form of the word.
- **Lexical Meanings**: List of distinct meanings associated with the lemma.
- **Usage Examples**: Sentences demonstrating how each meaning is used in context.

| Lemma | Gloss | Examples |
|-------|-------|----------|
| коса  | ['Заплетене волосся', 'Довге волосся'] | ['Якраз під старою вишнею стояла дівчина, хороша, як зоря ясна; руса коса нижче пояса', 'Очі в неї були великі, дві чорні коси, перекинуті наперед, обрамляли лице'] |
| коса  | ["Сільськогосподарське знаряддя для косіння трави, зернових, що має форму вузького зігнутого леза, прикріпленого до кісся дерев'яного держака"] | ['Свідок слави, дідівщини З вітром розмовляє, А внук косу несе в росу, За ними співає', 'Косарі косять, А вітер повіває, Шовкова трава На коси полягає'] |
| коса  | ['Піщана вузька, довга частина суходолу, що відокремлює від відкритого моря бухту, озеро або затоку; мис'] | ['Човен повернув за гострий ріг піскуватої коси і вступив у Чорне море', 'Скільки оком скинеш – леліє Дніпро, вигинаючись помежи горами, тихо миючи піскуваті коси'] |
| коса  | ['Селезінка'] | ['Коса свиняча, що коло печінки, довгенька'] |
| коса  | ['Південноафриканський етнос, що належить до групи народів банту'] | ['За генеалогічними переказами, коса є нащадками легендарного вождя Коса, від імені якого й походить назва етносу', 'У 1886 році британський дослідник Георг Тіль видав збірку казок і байок коса'] |


### Train Dataset
## How to reproduce the results
### Step 1: Preparation
### Step 2: Train
### Step 3: Evaluation

## Citation


TODO: publish paper on papers with code

TODO: publish WSD eval dataset and link it