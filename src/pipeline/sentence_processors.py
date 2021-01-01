import numpy as np
from .resources import (POLARITIES, POS_TAGS, SENTILEX_POLARITIES, EMOTICON_POLARITIES, EMOJI_POLARITIES)


def process_extracted_features(tokens_features, resource_name, resource_elements_number, use_relative_values=False):

    raw_resources = np.array(tokens_features.get(resource_name, np.zeros(resource_elements_number)))
    resources_count = np.sum(raw_resources, axis=0)
    resources_count[np.isnan(resources_count)] = 0.

    if use_relative_values:
        total = np.sum(resources_count) + 1e-7  # Avoid error in case of 0 resources count

        resources_percentage = resources_count / total
        resources_percentage[np.isnan(resources_percentage)] = 0.

        return [resources_percentage]
    else:
        return [resources_count]


def process_negative_words_features(tokens_features, use_relative_values=False):

    return process_extracted_features(tokens_features, 'process_negative_words', 1, use_relative_values)


def process_polarity_features(tokens_features, use_relative_values=False):

    return process_extracted_features(tokens_features, 'process_token_polarity', len(POLARITIES), use_relative_values)


def process_pos_tag_features(tokens_features, use_relative_values=False):

    return process_extracted_features(tokens_features, 'process_token_pos_tag', len(POS_TAGS), use_relative_values)


def process_sentilex_polarity_features(tokens_features, use_relative_values=False):

    return process_extracted_features(tokens_features, 'process_sentilex_word_polarity', len(SENTILEX_POLARITIES),
                                      use_relative_values)


def process_emoticon_polarity_features(tokens_features, use_relative_values=False):

    return process_extracted_features(tokens_features, 'process_emoticon_polarity', len(EMOTICON_POLARITIES),
                                      use_relative_values)


def process_emoji_polarity_features(tokens_features, use_relative_values=False):

    return process_extracted_features(tokens_features, 'process_emoji_polarity', len(EMOJI_POLARITIES),
                                      use_relative_values)
