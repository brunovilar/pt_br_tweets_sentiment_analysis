import numpy as np
import regex as re
from .. import settings
from .resources import (POLARITIES, POS_TAGS, SENTILEX_POLARITIES,
                        load_sentilex, load_emoji_polarity_index)
from .regular_expressions import EXTRACT_POLARITY


def process_token_text(token):
    return token.text


def process_token_lemma(token):
    return token.lemma if token.lemma else token.text


def polarity_extractor(token):
    if token.feats and (polarity_search := re.search(EXTRACT_POLARITY, token.feats)):
        return polarity_search.group(1)
    return None


def process_token_polarity(token):
    token_polarity = polarity_extractor(token)

    return [int(token_polarity == polarity) for polarity in POLARITIES]


def process_token_pos_tag(token):
    return [int(token.pos_tag == tag) for tag in POS_TAGS]


def process_negative_words(token):
    return [int(token.text in settings.NEGATIONS_WORDS)]


def process_sentilex_word_polarity(token):
    sentilex = load_sentilex()

    def get_sentilex_polarity_value(token):
        return sentilex.get(token.text) or sentilex.get(token.lemma) or '0'

    word_polarity = get_sentilex_polarity_value(token)

    return [int(word_polarity == polarity) for polarity in SENTILEX_POLARITIES]


def process_emoticon_polarity(token):
    if token.text in settings.NEGATIVE_EMOTICONS:
        return [1, 0]
    if token.text in settings.POSITIVE_EMOTICONS:
        return [0, 1]

    return [0, 0]


def process_emoji_polarity(token):
    emoji_sentiment = load_emoji_polarity_index()
    polarity_index = emoji_sentiment.get(token.text, -1)
    return (np.array([0, 1, 2]) == polarity_index).astype(int)


def process_token_pos_tag(token):
    return [int(token.pos_tag == tag) for tag in POS_TAGS]
