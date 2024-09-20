from services.utils_model import run_inference
from collections import Counter


def _find_target_word_in_sentence(udpipe_model, input_text, target_word):
    """
    TODO: This function wil only find one target word
    """

    # TODO fix misunderstandin issue with similarity of tokenize and tokenize_text
    tokenized = udpipe_model.tokenize(input_text)

    for tok_sent in tokenized:

        udpipe_model.tag(tok_sent)

        for word_index, w in enumerate(tok_sent.words[1:]):  # under 0 index is root
            # print(w.lemma)

            if w.lemma.lower() == target_word.lower():
                # TODO check if this better return tok_sent.words[word_index+1].form
                return [w.form for w in tok_sent.words[1:]][word_index]
    return None


def _find_target_word_in_tokenized_text(tokenizer, tokenized_input_text, word):
    # TODO check if better list(set(tokenized_input_text.word_ids()))
    unique_w_ids = list(Counter(tokenized_input_text.word_ids()).keys())
    unique_w_ids.remove(None)
    tokens = ["" for _ in range(len(unique_w_ids))]

    word_indexes = []
    prev_word_id = None

    target_words_with_indexes = []

    for index, (input_id, word_id) in enumerate(zip(tokenized_input_text["input_ids"][0],
                                                    tokenized_input_text.word_ids())):
        token = tokenizer.decode([input_id]).strip()

        if prev_word_id == word_id:
            word_indexes.append(index)

        else:
            word_indexes = []
            word_indexes.append(index)

        prev_word_id = word_id

        if word_id is None:
            continue

        if token.startswith("##"):
            token = token.replace("##", "")

        tokens[word_id] += token
        if tokens[word_id].replace("▁", "").lower() == word.lower():
            target_words_with_indexes.append((word, word_indexes.copy()))
    # print(tokens)

    return target_words_with_indexes


def get_target_word_embedding(model, tokenizer, udpipe_model, pooling_strategy, target_word, sentence_example, device):
    # TODO: remove it
    ACUTE = chr(0x301)
    GRAVE = chr(0x300)
    target_word_fixed = target_word.replace(GRAVE, "").replace(ACUTE, "")

    # TODO rename "word" for better understanding
    word = _find_target_word_in_sentence(udpipe_model, sentence_example, target_word_fixed)
    if word is None:
        # print("Can't find target word in sentence")
        return None

    tokenized_input_text, hidden_states = run_inference(model, tokenizer, sentence_example, device)
    target_word_indexes = _find_target_word_in_tokenized_text(tokenizer, tokenized_input_text, word)

    # TODO if len(target_word_indexes) != 1:
    if len(target_word_indexes) == 0:
        # print("Cant find target word in tokens")
        return None
    # TODO we drop all word if there are more that 2 target word occurence in example - its bad
    if len(target_word_indexes) > 1:
        # print("Skip for now")
        return None

    target_word_indexes = target_word_indexes[0][1]  # TODO: explain
    return pooling_strategy(hidden_states[target_word_indexes[0]:target_word_indexes[-1] + 1])


def get_context_embedding(model, tokenizer, pooling_strategy, context, device):
    _, hidden_states_context = run_inference(model, tokenizer, context, device)
    return pooling_strategy(hidden_states_context)
