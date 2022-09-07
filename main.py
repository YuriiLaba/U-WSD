# import pandas as pd
from src.utils_data import read_and_transform_data, prepare_frequent_dictionary
# from src.utils_results import results_reports
# from src.word_sense_detector import WordSenseDetector
# from src.udpipe_model import UDPipeModel

# udpipe_model = UDPipeModel("drive/MyDrive/Colab Notebooks/ukr_nlp/20180506.uk.mova-institute.udpipe")
# word_sense_detector = WordSenseDetector("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", udpipe_model, evaluation_dataset_pd)
# evaluation_dataset_pd = word_sense_detector.run()

data = read_and_transform_data('data/sum_12_full_try6.jsonlines')

# prepare_frequent_dictionary('data/ubertext.fiction_news_wikipedia.filter_rus+short.csv.xz')
# evaluation_dataset_pd = pd.read_csv('data/dummy_prediction.csv')
# results_reports(evaluation_dataset_pd)
