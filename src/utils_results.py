import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from src.utils_data import take_first_n_glosses, add_pos_tag, add_frequency_column
from src.config import MINIMUM_POS_OCCURRENCE, MINIMUM_GLOSS_OCCURRENCE, FREQUENCY_QUANTILES, ACUTE


def prediction_accuracy(data_with_predictions):
    data_dropna = data_with_predictions.dropna()

    return accuracy_score(data_dropna["gloss"], data_dropna["predicted_context"])


def prediction_error(data_with_predictions):
    wrong_prediction = data_with_predictions[
        data_with_predictions['gloss'] != data_with_predictions['predicted_context']]
    wrong_prediction.drop(columns=['clear_lemma'], errors='ignore')
    wrong_prediction.to_csv('badly_predicted.csv', index=False)


def generate_results_by_pos(data_with_predictions, udpipe_model):
    if 'pos' not in data_with_predictions.columns:
        data_with_predictions = add_pos_tag(data_with_predictions, udpipe_model)

    results = {'overall_accuracy': [prediction_accuracy(data_with_predictions), len(data_with_predictions)]}

    for pos in data_with_predictions["pos"].value_counts().iteritems():
        if pos[1] < MINIMUM_POS_OCCURRENCE:
            break
        data_with_predictions_pos = data_with_predictions[data_with_predictions["pos"] == pos[0]]
        results[pos[0] + '_accuracy'] = [prediction_accuracy(data_with_predictions_pos), len(data_with_predictions_pos)]
    results_df = pd.DataFrame.from_dict(results,
                                        orient='index',
                                        columns=['accuracy', 'count'])
    return results_df


def gef_number_of_gloss_for_lemma(data):
    return data.groupby('lemma').gloss.count()


def generate_results_by_count_of_gloss(data_with_predictions):
    gloss_count = gef_number_of_gloss_for_lemma(data_with_predictions).to_dict()
    data_with_predictions['gloss_count'] = data_with_predictions['lemma'].map(gloss_count)

    results = {'overall_accuracy': [prediction_accuracy(data_with_predictions), len(data_with_predictions)]}

    for gloss_count in data_with_predictions["gloss_count"].value_counts().sort_index().iteritems():
        if gloss_count[1] < MINIMUM_GLOSS_OCCURRENCE:
            data_with_predictions_gloss = data_with_predictions[data_with_predictions["gloss_count"] >= gloss_count[0]]
            results[f'word_that_have_{gloss_count[0]}+_glosses'] = [prediction_accuracy(data_with_predictions_gloss),
                                                                    len(data_with_predictions_gloss)]
            break
        data_with_predictions_gloss = data_with_predictions[data_with_predictions["gloss_count"] == gloss_count[0]]
        results[f'word_that_have_{gloss_count[0]}_glosses'] = [prediction_accuracy(data_with_predictions_gloss),
                                                               len(data_with_predictions_gloss)]
    results_df = pd.DataFrame.from_dict(results,
                                        orient='index',
                                        columns=['accuracy', 'count'])
    return results_df


def results_by_lemma_order(data_with_predictions):
    data_with_predictions['lemma_order'] = data_with_predictions.groupby("lemma").cumcount() + 1
    results = data_with_predictions.groupby('lemma_order').apply(prediction_accuracy)
    results.name = 'accuracy'
    return results


def generate_results_filter_gloss_frequency(data_with_predictions):
    results = {'overall_accuracy': [prediction_accuracy(data_with_predictions), len(data_with_predictions)]}

    max_gloss_for_lemma = gef_number_of_gloss_for_lemma(data_with_predictions).max()
    for i in range(1, max_gloss_for_lemma+1):
        data_with_predictions_filtered_gloss = take_first_n_glosses(data_with_predictions, i)
        results[f'take first {i} gloss'] = [prediction_accuracy(data_with_predictions_filtered_gloss),
                                            len(data_with_predictions_filtered_gloss)]
    results_df = pd.DataFrame.from_dict(results,
                                        orient='index',
                                        columns=['cum_accuracy', 'cum_count'])

    results_df = results_df.join(results_by_lemma_order(data_with_predictions))

    return results_df


def generate_results_filter_lemma_frequency(data_with_predictions, udpipe_model):
    results = {'overall_accuracy': [prediction_accuracy(data_with_predictions), len(data_with_predictions)]}

    if 'freq_in_corpus' not in data_with_predictions.columns:
        data_with_predictions = add_frequency_column(data_with_predictions, udpipe_model)

    quantiles = [round(i * 1/FREQUENCY_QUANTILES, 1) for i in range(1, FREQUENCY_QUANTILES)]
    quantile_points = np.quantile(data_with_predictions['freq_in_corpus'], quantiles)

    for quantile, quantile_point in zip(quantiles, quantile_points):
        data_with_predictions_filtered_lemma_freq = data_with_predictions[
            data_with_predictions['freq_in_corpus'] > quantile_point]
        results[f'remove lower {quantile * 100}% freq'] = [
            prediction_accuracy(data_with_predictions_filtered_lemma_freq),
            len(data_with_predictions_filtered_lemma_freq)]
    results_df = pd.DataFrame.from_dict(results,
                                        orient='index',
                                        columns=['accuracy', 'count'])
    return results_df


def results_reports(data_with_predictions, udpipe_model, frequency_report=False):
    print('Accuracy by part of lang')
    print(generate_results_by_pos(data_with_predictions, udpipe_model), '\n')
    print('Accuracy by number of gloss for word')
    # TODO fix to avoid 1 gloss lemmas
    print(generate_results_by_count_of_gloss(data_with_predictions), '\n')
    if frequency_report:
        print('Accuracy by lemma frequency')
        print(generate_results_filter_lemma_frequency(data_with_predictions, udpipe_model), '\n')
    print('Accuracy by taking first n gloss')
    # TODO add here accumulated accuracy and separate for each lemma
    print(generate_results_filter_gloss_frequency(data_with_predictions), '\n')
    prediction_error(data_with_predictions)


def prediction_comparison(base, new):
    print(f"Total data shape = {base.shape[0]}")
    accuracy_base = prediction_accuracy(base)
    accuracy_new = prediction_accuracy(new)

    print(f"Accuracy base = {round(accuracy_base*100, 2)}%")
    print(f"Accuracy new = {round(accuracy_new*100, 2)}%")
    print(f"Improved by {round((accuracy_new - accuracy_base) * 100, 3)}%, its {round((accuracy_new - accuracy_base) * base.shape[0], 0)} samples")
    correct_base_miss_new = base[(base.gloss == base.predicted_context) & (new.gloss != new.predicted_context)]
    correct_new_miss_base = new[(base.gloss != base.predicted_context) & (new.gloss == new.predicted_context)]

    print(f"Correct in base missed in new = {correct_base_miss_new.shape[0]}")
    print(f"Correct in best missed in baseline = {correct_new_miss_base.shape[0]}")
    return correct_base_miss_new, correct_new_miss_base
