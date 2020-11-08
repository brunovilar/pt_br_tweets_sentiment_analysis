import numpy as np
import funcy as fp
from collections import defaultdict
from collections.abc import Iterable
from .adaptors import get_adaptor_fn
from .processors import process_word_lemma


def extract_tokens_and_features(text, nlp_pipeline, word_processors=None, sentence_processors=None):

    if not isinstance(text, str) and isinstance(text, Iterable):
        text = fp.first(text)

    if not word_processors:
        word_processors = []

    if not sentence_processors:
        sentence_processors = []

    tokens = []
    word_features = defaultdict(list)
    sentence_features = {}

    adaptor_fn = get_adaptor_fn(nlp_pipeline)
    document = adaptor_fn(nlp_pipeline(text))

    for token in document:
        tokens.append(process_word_lemma(token))

        for processor_fn in word_processors:
            word_features[processor_fn.__name__].append(processor_fn(token))

    for processor_fn in sentence_processors:
        sentence_features[processor_fn.__name__] = processor_fn(word_features)

    features = fp.lflatten([sentence_features[item.__name__] for item in sentence_processors])
    features = np.concatenate(features, axis=0)

    return tokens, features


def extract_features(*args, **kwargs):

    _, features = extract_tokens_and_features(*args, **kwargs)

    return features


def extract_tokens(text, nlp_pipeline):

    if not isinstance(text, str) and isinstance(text, Iterable):
        text = fp.first(text)

    adaptor_fn = get_adaptor_fn(nlp_pipeline)
    document = adaptor_fn(nlp_pipeline(text))

    tokens = []
    for token in document:
        tokens.append(process_word_lemma(token))

    return tokens
