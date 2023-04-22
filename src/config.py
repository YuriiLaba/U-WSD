# [PREPARATION]
MIN_LEMMA_LENTH = 3
MAX_GLOSS_OCCURRENCE = 4
ACUTE = chr(0x301)
GRAVE = chr(0x300)
LEMMAS_TO_REMOVE = ['або', 'ага', 'адже', 'але', 'ану', 'ані', 'вона', 'еге', 'летяга', 'лише', 'мирно', 'немовби',
                    'нерозкладний', 'нехай', 'ніби', 'нібито', 'отже', "коли", "отож", "геть", "єсть"]

# [RESULTS]
MINIMUM_POS_OCCURRENCE = 100
MINIMUM_GLOSS_OCCURRENCE = 300
FREQUENCY_QUANTILES = 10

# [DATA MINING]
PATH_TO_SOURCE_DATASET = "all_uniq_filtered_shuffled.txt.bz2"
PATH_TO_SOURCE_UDPIPE = "20180506.uk.mova-institute.udpipe"
PATH_TO_SAVE_GATHERED_DATASET = "lemma_examples.json"
PATH_TO_LEMMAS_OF_INTEREST = "unique_lemmas_homonyms.txt"
NUMBER_OF_EXAMPLES_TO_GATHER = 5_000_000
