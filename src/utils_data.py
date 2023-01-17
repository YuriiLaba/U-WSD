import lzma
import pandas as pd
import pymorphy2
import os
import re
import stanza
import configparser

config = configparser.ConfigParser()
config.read('src/config.ini')

MIN_LEMMA_LENTH = config.getint('PREPARATION', 'MIN_LEMMA_LENTH')
MAX_GLOSS_OCCURRENCE = config.getint('PREPARATION', 'MAX_GLOSS_OCCURRENCE')

ACUTE = chr(0x301)
GRAVE = chr(0x300)


def take_first_n_glosses(data, first_n_glosses):
    data = data.groupby('lemma').head(first_n_glosses)
    return data


def read_and_transform_data(path, homonym=False, gloss_strategy='first'):
    data = pd.read_json(path, lines=True).drop(columns=['suffixes', 'tags', 'phrases', 'word_id', 'url'])
    data = data[data.lemma.apply(len) > MIN_LEMMA_LENTH]

    data.dropna(subset=['synsets'], inplace=True)
    data = data[data.synsets.apply(len) > 0]

    if homonym:
        data['lemma'] = data.lemma.apply(lambda x: x.lower().replace(GRAVE, "").replace(ACUTE, ""))
        data = data.groupby("lemma").filter(lambda x: len(x) > 1)
        data['synsets'] = data['synsets'].apply(lambda x: x[0])
    else:
        data = data.explode('synsets')

    data = data[data['synsets'].apply(lambda x: len(x['gloss'])) > 0]
    data = data[data['synsets'].apply(lambda x: len(x['examples'])) > 0]

    data = pd.concat([data.lemma, data.synsets.apply(pd.Series)], axis=1)
    data.drop(columns=['sense_id'], inplace=True)

    if gloss_strategy == 'first':
        data['gloss'] = data['gloss'].apply(lambda x: x[0])
    elif gloss_strategy == 'concat':
        data['gloss'] = data['gloss'].apply(lambda x: '. '.join(x))

    data['examples'] = data['examples'].apply(lambda x: [i["ex_text"] for i in x])
    gloss_to_remove = data.groupby("gloss").filter(lambda x: len(x) > MAX_GLOSS_OCCURRENCE)["gloss"].tolist()
    data = data[~data["gloss"].isin(gloss_to_remove)]

    data = data.groupby("lemma").filter(lambda x: len(x) > 1)

    pattern = r' \([^\(]*?знач[^\)]*?\)'
    data.gloss = data.gloss.apply(lambda x: re.sub(pattern, '', x))
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

    data[['freq_by_pos', 'freq_in_corpus', 'doc_frequency']] = data[['freq_by_pos', 'freq_in_corpus', 'doc_frequency']].fillna(0)

    del dictionary, data_not_merged
    return data
