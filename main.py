import pandas as pd
from src.utils_data import read_and_transform_data, prepare_frequent_dictionary, add_frequency_column
from src.utils_results import results_reports

# data = read_and_transform_data('data/sum_12_full_try6.jsonlines')

# predicted_data = prediction_function(data)
predicted_data = pd.read_csv('data/dummy_prediction.csv')

# TODO add checking if freq_dictionary exist and run only if not exist
# prepare_frequent_dictionary('data/ubertext.fiction_news_wikipedia.filter_rus+short.csv.xz')

results_reports(predicted_data)
