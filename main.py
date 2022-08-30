import pandas as pd
from src.utils_data import read_and_transform_data
from src.utils_results import results_reports

# data = read_and_transform_data('data/sum_12_full_try6.jsonlines')
# predicted_data = prediction_function(data)
predicted_data = pd.read_csv('data/dummy_prediction.csv')
results_reports(predicted_data)
