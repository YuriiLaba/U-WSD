import pandas as pd
from sklearn.metrics import accuracy_score
import pymorphy2


def prediction_accuracy(data_with_predictions):
    data_dropna = data_with_predictions.dropna()

    return accuracy_score(data_dropna["gloss"], data_dropna["predicted_context"])


def prediction_error(data_with_predictions):
    wrong_prediction = data_with_predictions[
        data_with_predictions['gloss'] != data_with_predictions['predicted_context']]
    wrong_prediction[['lemma', 'gloss', 'predicted_context']].to_csv('badly_predicted.csv', index=False)


def add_pos_tag(data_with_predictions):
    morph = pymorphy2.MorphAnalyzer(lang='uk')

    def get_pos_tag(row):
        p = morph.parse(row["lemma"])[0]
        return p.tag.POS

    data_with_predictions["pos"] = data_with_predictions.apply(get_pos_tag, axis=1)
    return data_with_predictions


def generate_results_df(data_with_predictions):
    data_with_predictions = add_pos_tag(data_with_predictions)
    data_with_predictions_nouns = data_with_predictions[data_with_predictions["pos"] == "NOUN"]
    data_with_predictions_verb = data_with_predictions[data_with_predictions["pos"] == "VERB"]
    results_df = pd.DataFrame.from_dict({'overall_accuracy': prediction_accuracy(data_with_predictions),
                                         'noun_accuracy': prediction_accuracy(data_with_predictions_nouns),
                                         'verb_accuracy': prediction_accuracy(data_with_predictions_verb)},
                                        orient='index',
                                        columns = ['accuracy'])
    return results_df
