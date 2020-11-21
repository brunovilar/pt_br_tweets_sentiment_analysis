import numpy as np
import regex as re
from .. import settings
from .resources import (POLARITIES, POS_TAGS, SENTILEX_POLARITIES,
                        load_sentilex, load_emoji_polarity_index)
from .regular_expressions import EXTRACT_POLARITY


def process_word_text(word):
    return word.text


def process_word_lemma(word):
    return word.lemma if word.lemma else word.text


def polarity_extractor(word):
    if word.feats and (polarity_search := re.search(EXTRACT_POLARITY, word.feats)):
        return polarity_search.group(1)
    return None


def process_word_polarity(word):
    word_polarity = polarity_extractor(word)

    return [int(word_polarity == polarity) for polarity in POLARITIES]


def process_word_pos_tag(word):
    return [int(word.pos_tag == tag) for tag in POS_TAGS]


def process_negative_words(word):
    return [int(word.text in settings.NEGATIONS_WORDS)]


def process_sentilex_word_polarity(word):
    sentilex = load_sentilex()

    def get_sentilex_polarity_value(word):
        return sentilex.get(word.text) or sentilex.get(word.lemma) or '0'

    word_polarity = get_sentilex_polarity_value(word)

    return [int(word_polarity == polarity) for polarity in SENTILEX_POLARITIES]


def process_emoticon_polarity(word):
    if word.text in settings.NEGATIVE_EMOTICONS:
        return [1, 0]
    if word.text in settings.POSITIVE_EMOTICONS:
        return [0, 1]

    return [0, 0]


def process_emoji_polarity(word):
    emoji_sentiment = load_emoji_polarity_index()
    polarity_index = emoji_sentiment.get(word.text, -1)
    return (np.array([0, 1, 2]) == polarity_index).astype(int)


def process_word_pos_tag(word):
    return [int(word.pos_tag == tag) for tag in POS_TAGS]
