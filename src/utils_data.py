import pandas as pd

MIN_LEMMA_LENTH = 3
MAX_GLOSS_OCCURRENCE = 1


def filter_gloss_frequency(data, max_gloss_frequency):
    data = data.groupby('lemma').head(max_gloss_frequency)
    return data


def read_and_transform_data(path):
    data = pd.read_json(path, lines=True)

    data = data[data.lemma.apply(len) > MIN_LEMMA_LENTH]

    data = data.explode('synsets')
    data.dropna(subset=['synsets'], inplace=True)

    data = data[data['synsets'].apply(lambda x: len(x['gloss'])) == 1]
    data = data[data['synsets'].apply(lambda x: len(x['examples'])) > 0]

    data = pd.concat([data.lemma, data.synsets.apply(pd.Series)], axis=1)
    data.drop(columns=['sense_id'], inplace=True)
    data['gloss'] = data['gloss'].apply(lambda x: x[0])
    data['examples'] = data['examples'].apply(lambda x: x[0]['ex_text'])

    gloss_to_remove = data.groupby("gloss").filter(lambda x: len(x) > MAX_GLOSS_OCCURRENCE)["gloss"].tolist()
    data = data[~data["gloss"].isin(gloss_to_remove)]

    data = data.groupby("lemma").filter(lambda x: len(x) > 1)
    return data
