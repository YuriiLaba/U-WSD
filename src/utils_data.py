import lzma
import pandas as pd
import pymorphy2
import os
import re
import stanza
from src.config import MIN_LEMMA_LENTH, MAX_GLOSS_OCCURRENCE, ACUTE, GRAVE, LEMMAS_TO_REMOVE
from functools import reduce
import operator


def take_first_n_glosses(data, first_n_glosses):
    data = data.groupby('lemma').head(first_n_glosses)
    return data


def clean_badly_parsed_data(data):
    # patterns_to_clear = ["(?i)Те саме[ ,]+[0-9. ,що;–)]+",
    #                      "(?i)дія за знач[0-9. ,і;–)]+",
    #                      "(?i)стан за знач[0-9. ,і;–)]+",
    #                      "(?i)Прикм. до[0-9. ,і;–)]+",
    #                      "(?i)Зменш. до[0-9. ,і;–)]+",
    #                      "(?i)Вищ. ст. до[0-9. ,і;–)]+",
    #                      "(?i)Док. до[0-9. ,і;–)]+",
    #                      "(?i)Присл. до[0-9. ,і;–)]+",
    #                      " . . [0-9 ,\)–]+"
    #                      ]
    # for pattern in patterns_to_clear:
    #     data.gloss = data.gloss.apply(lambda x: re.sub(pattern, '', x))
    # data = data[data['gloss'].apply(len) > 2]
    # data = data.groupby("lemma").filter(lambda x: len(x) > 1)

    replace_short = {"Вигот.": "Виготовлений",
                     "Власт.": "Властивий",
                     "Признач.": "Призначений",
                     "Зробл.": "Зроблений",
                     "і т. ін.": ""}

    for r in replace_short:
        data.gloss = data.gloss.str.replace(r, replace_short[r])

    return data


def remove_sense_reference(data):
    patterns_to_clear = ['Абстр. ім.',
                         'Вищ. ст. до',
                         'Док. до',
                         'Дія за знач.',
                         'Дієпр. акт',
                         'Дієпр. пас.',
                         'Жін. до',
                         'Збільш. до',
                         'Зменш. до',
                         'Зменш.-пестл. до',
                         'Однокр. до',
                         'Пас. до',
                         'Пестл. до',
                         'Прикм. до',
                         'Присл. до',
                         'Підсил. до',
                         'Стан за знач.',
                         'Стос. до',
                         'Те саме',
                         'дія за знач',
                         'стан за знач']

    def check_if_drop(row):
        for i in patterns_to_clear:
            if i in row:
                return True
        return False

    lemmas_to_drop = data[data["gloss"].apply(check_if_drop)]["lemma"].values
    return data[~data["lemma"].isin(lemmas_to_drop)]


def homonym_preparation(data):
    data['lemma'] = data.lemma.apply(lambda x: x.lower().replace(GRAVE, "").replace(ACUTE, ""))
    data = data.groupby("lemma").filter(lambda x: len(x) > 1)
    data['order'] = data.groupby('lemma', as_index=False)['lemma'].cumcount()
    return data


def drop_duplicates(data):
    data.sort_values('lemma', inplace=True)
    data.drop_duplicates(subset=['lemma', 'gloss'], inplace=True)

    data['str_examples'] = data.examples.astype(str)
    data.drop_duplicates(subset=['str_examples', 'gloss'], inplace=True)
    data.drop(columns=['str_examples'], inplace=True)
    return data


def parse_synset(data):
    data = data.explode('synsets')
    data = data[data['synsets'].apply(lambda x: len(x['gloss'])) > 0]
    data = data[data['synsets'].apply(lambda x: len(x['examples'])) > 0]
    data = pd.concat([data.drop(columns=['synsets']), data.synsets.apply(pd.Series)], axis=1)
    data.drop(columns=['sense_id'], inplace=True)
    data['examples'] = data['examples'].apply(lambda x: [i["ex_text"] for i in x])
    return data


def read_and_transform_data(path, homonym=False, gloss_strategy='first', remove_reference_lemma=True):
    data = pd.read_json(path, lines=True).drop(columns=['suffixes', 'tags', 'phrases', 'word_id', 'url', 'prime'])
    data = data[data.lemma.apply(len) > MIN_LEMMA_LENTH]

    data.dropna(subset=['synsets'], inplace=True)
    data = data[data.synsets.apply(len) > 0]

    if homonym:
        data = homonym_preparation(data)

    data = parse_synset(data)

    if gloss_strategy == 'first':
        data['gloss'] = data['gloss'].apply(lambda x: x[0])
    elif gloss_strategy == 'concat':
        data['gloss'] = data['gloss'].apply(lambda x: '. '.join(x))

    gloss_to_remove = data.groupby("gloss").filter(lambda x: len(x) > MAX_GLOSS_OCCURRENCE)["gloss"].tolist()
    data = data[~data["gloss"].isin(gloss_to_remove)]

    data = clean_badly_parsed_data(data)

    if remove_reference_lemma:
        data = remove_sense_reference(data)

    data = data[~data.lemma.isin(LEMMAS_TO_REMOVE)]

    data = drop_duplicates(data)

    pattern = r' \([^\(]*?знач[^\)]*?\)'
    data.gloss = data.gloss.apply(lambda x: re.sub(pattern, '', x))

    if homonym:
        data = data.groupby(['lemma', 'order']).agg({"gloss": list,
                                                     "examples": lambda x: reduce(operator.concat, x)}).reset_index().drop(columns=['order'])
    else:
        data['gloss'] = data.gloss.apply(lambda x: [x])

    data = data.groupby("lemma").filter(lambda x: len(x) > 1)
    return data


def prepare_frequent_dictionary(path, force_rebuild=False, save_errors=False):
    if os.path.exists('data/frequents.pkl') and not force_rebuild:
        print('Frequency df already exist, use force_rebuild=True to rebuild it')
        return

    lines_of_file = []
    errors_lines = []

    # TODO think how to correct parse dictionary to avoid errors

    with open(path, 'rb') as compressed:
        with lzma.LZMAFile(compressed) as uncompressed:
            for line in uncompressed:
                parsed_line = line.decode('utf8')[:-2].split(',')
                if len(parsed_line) != 7:
                    if save_errors:
                        errors_lines.append(line)
                    continue
                lines_of_file.append(parsed_line)

    df = pd.DataFrame(lines_of_file[1:], columns=lines_of_file[0])
    df.lemma = df.lemma.str.replace("’", "'").str.lower()
    for numeric_col in ['count', 'doc_count', 'freq_by_pos', 'freq_in_corpus', 'doc_frequency']:
        df[numeric_col] = df[numeric_col].astype('float')

    df = df.groupby(['lemma', 'pos']).sum().reset_index()

    if not os.path.exists('data'):
        os.mkdir('data')

    df.sort_values(['lemma', 'freq_in_corpus'], inplace=True)
    df.to_pickle('data/frequents.pkl')

    if save_errors:
        df = pd.DataFrame(errors_lines, columns=["column"])
        df.to_csv('data/errors_of_dict.csv', index=False)

    del df


def add_pos_tag(data_with_predictions, udpipe_model=None, engine="stanza"):
    if engine == 'stanza':
        if os.path.exists('data/pos_precalculation.pkl'):
            pos_precalculation = pd.read_pickle('data/pos_precalculation.pkl')
            data_with_predictions.loc[:, "pos"] = data_with_predictions.lemma.replace(pos_precalculation['pos'])
        else:
            nlp = stanza.Pipeline(lang='uk', processors='tokenize,mwt,pos', verbose=False)

            def get_pos_tag_udpipe(word, text):
                word = word.lower().replace(GRAVE, "").replace(ACUTE, "").replace("'", "’")
                tokens = udpipe_model.tokenize(text)
                for tok_sent in tokens:
                    udpipe_model.tag(tok_sent)
                    for word_index, w in enumerate(tok_sent.words[1:]):
                        if w.lemma == word:
                            return w.upostag
                doc = nlp(word)
                return doc.sentences[0].words[0].upos

            data_with_predictions["pos"] = data_with_predictions.apply(lambda x: get_pos_tag_udpipe(x['lemma'],
                                                                                                    x['examples'][0]),
                                                                       axis=1)
            if not os.path.exists("data"):
                os.mkdir('data')
            data_with_predictions[['lemma', 'pos']].set_index('lemma').to_pickle('data/pos_precalculation.pkl')
    elif engine == 'pymorphy':
        morph = pymorphy2.MorphAnalyzer(lang='uk')

        def get_pos_tag(row):
            p = morph.parse(row["lemma"])[0]
            return p.tag.POS

        data_with_predictions["pos"] = data_with_predictions.apply(get_pos_tag, axis=1)

    return data_with_predictions


def add_frequency_column(data, udpipe_model):
    dictionary = pd.read_pickle('data/frequents.pkl')

    if 'pos' not in data.columns:
        data = add_pos_tag(data, udpipe_model)

    data['clear_lemma'] = data.lemma.str.replace(GRAVE, "").str.replace(ACUTE, "").str.lower()
    important_columns = data.columns

    data = data.merge(dictionary,
                      how='left',
                      left_on=['clear_lemma', 'pos'],
                      right_on=['lemma', 'pos'],
                      suffixes=('', '_y'))

    data_not_merged = data[data.freq_in_corpus.isna()][important_columns].copy()
    data = data[data.freq_in_corpus.notna()]

    dictionary.drop_duplicates(subset=['lemma'], keep='last', inplace=True)
    data_not_merged = data_not_merged.merge(dictionary.drop(columns=['pos']),
                                            how='left',
                                            left_on=['clear_lemma'],
                                            right_on=['lemma'],
                                            suffixes=('', '_y')
                                            )

    data = pd.concat([data, data_not_merged])
    data.drop(columns=['clear_lemma', 'lemma_y', 'count', 'doc_count', 'pos_y'], inplace=True, errors='ignore')

    data[['freq_by_pos', 'freq_in_corpus', 'doc_frequency']] = data[
        ['freq_by_pos', 'freq_in_corpus', 'doc_frequency']].fillna(0)

    del dictionary, data_not_merged
    return data
