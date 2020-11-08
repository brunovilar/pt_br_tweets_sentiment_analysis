import numpy as np
from .resources import (POLARITIES, POS_TAGS, SENTILEX_POLARITIES)

def compute_negative_words_features(words_features):
    counts = np.array(words_features.get('process_negative_words', [0]))
    features_count = np.sum(counts)

    return np.array([features_count])


def compute_polarity_features(words_features):
    words_polarity = np.array(words_features.get('process_word_polarity', np.zeros(len(POLARITIES))))
    polarities_count = np.sum(words_polarity, axis=0)
    total = np.sum(polarities_count)

    # return [polarities_count, polarities_count / total]
    return [polarities_count]


def compute_pos_tag_features(words_features):
    counts = np.array(words_features.get('process_word_pos_tag', np.zeros(len(POS_TAGS))))
    features_count = np.sum(counts, axis=0)
    features_count[np.isnan(features_count)] = 0.
    total = np.sum(features_count)
    total = 1 if total == 0 else total

    # return [features_count, features_count / total]
    return [features_count]


def compute_sentilex_polarity_features(words_features):
    words_polarity = np.array(words_features.get('process_sentilex_word_polarity', np.zeros(len(SENTILEX_POLARITIES))))
    polarities_count = np.sum(words_polarity, axis=0)
    total = np.sum(polarities_count)

    # return [polarities_count, polarities_count / total]
    return [polarities_count]


def compute_emoticon_polarity_features(words_features):
    words_polarity = np.array(words_features.get('process_emoticon_polarity'))
    polarities_count = np.sum(words_polarity, axis=0)
    total = np.sum(polarities_count)

    polarities_percentage = polarities_count / total
    polarities_percentage[np.isnan(polarities_percentage)] = 0.

    # return [polarities_count, polarities_percentage]
    return [polarities_count]


def compute_emoji_polarity_features(words_features):
    words_polarity = np.array(words_features.get('process_emoji_polarity'))
    polarities_count = np.sum(words_polarity, axis=0)
    polarities_count[np.isnan(polarities_count)] = 0.

    total = np.sum(polarities_count)

    polarities_percentage = polarities_count / total
    polarities_percentage[np.isnan(polarities_percentage)] = 0.

    # return [polarities_count, polarities_percentage]
    return [polarities_count]
