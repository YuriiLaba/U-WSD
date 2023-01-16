# word-sense-ambiguation

[СЛОВНИК УКРАЇНСЬКОЇ МОВИ](https://services.ulif.org.ua/expl/Entry)


### Data preparation

0. Prepare utilities files
   - Parse ubertext wiki dataset
   - Get freq_in_corpus for each pair lemma + pos
1. Read and transform data
   - Remove lemmas shorter than MIN_LEMMA_LENTH
   - Remove lemmas with missing synsets (АНТО́НІВ, БОРТНЯ́К т.п.)
   - Remove lemmas with missing gloss (ГИ́ДКО, ЛЯ́ШКА)
   - Remove lemmas with missing examples (ПЕРЕХОРУВА́ТИ, ОБЕЗКРО́ВИТИСЯ)
   - Take first part of gloss for each lemma (TODO experiment with other approaches: concat, take all parts separate)
   - Remove lemmas with gloss that appears more than MAX_GLOSS_OCCURRENCE
2. Take first n_senses glosses for each lemma


### Validation method
- Results by lemma POS
  - Get POS of lemma with pymorphy2
  - Remove low occurrence POS (occurrence in data set less than MINIMUM_POS_OCCURRENCE)
- Results by number of different gloss per lemma
  - Get result by number of different gloss for lemma.
  - Combine results for numbers of unique gloss per lemma less than MINIMUM_GLOSS_OCCURRENCE 
- Results by frequency of lemma
  - Merge dataframe with prepared frequency dataframe by lemma+pos if possible or with most frequent lemma+pos if missing
  - Calculate FREQUENCY_QUANTILES quantiles for frequency 
  - Iterative print results by removing lowest frequency quantiles
- Results by first n gloss for each lemma
  - Get results if we calculate accuracy only by first n gloss for lemmas

