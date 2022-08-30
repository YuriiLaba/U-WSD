import pandas as pd
from sklearn.metrics import accuracy_score
import pymorphy2

MINIMUM_POS_OCCURRENCE = 100
MINIMUM_GLOSS_OCCURRENCE = 300


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


def generate_results_by_pos(data_with_predictions):
    data_with_predictions = add_pos_tag(data_with_predictions)

    results = {'overall_accuracy': [prediction_accuracy(data_with_predictions), len(data_with_predictions)]}

    for pos in data_with_predictions["pos"].value_counts().iteritems():
        if pos[1] < MINIMUM_POS_OCCURRENCE:
            break
        data_with_predictions_pos = data_with_predictions[data_with_predictions["pos"] == pos[0]]
        results[pos[0]+'_accuracy'] = [prediction_accuracy(data_with_predictions_pos), len(data_with_predictions_pos)]
    results_df = pd.DataFrame.from_dict(results,
                                        orient='index',
                                        columns=['accuracy', 'count'])
    return results_df


def generate_results_by_gloss(data_with_predictions):
    gloss_count = data_with_predictions.groupby('lemma').gloss.count().to_dict()
    data_with_predictions['gloss_count'] = data_with_predictions['lemma'].map(gloss_count)

    results = {'overall_accuracy': [prediction_accuracy(data_with_predictions), len(data_with_predictions)]}

    for gloss_count in data_with_predictions["gloss_count"].value_counts().sort_index().iteritems():
        if gloss_count[1] < MINIMUM_GLOSS_OCCURRENCE:
            data_with_predictions_gloss = data_with_predictions[data_with_predictions["gloss_count"] >= gloss_count[0]]
            results[str(gloss_count[0]) + '+_gloss_word_accuracy'] = [prediction_accuracy(data_with_predictions_gloss),
                                                                      len(data_with_predictions_gloss)]
            break
        data_with_predictions_gloss = data_with_predictions[data_with_predictions["gloss_count"] == gloss_count[0]]
        results[str(gloss_count[0])+'_gloss_word_accuracy'] = [prediction_accuracy(data_with_predictions_gloss),
                                                               len(data_with_predictions_gloss)]
    results_df = pd.DataFrame.from_dict(results,
                                        orient='index',
                                        columns=['accuracy', 'count'])
    return results_df


def results_reports(data_with_predictions):
    print(generate_results_by_pos(data_with_predictions))
    print(generate_results_by_gloss(data_with_predictions))
