import funcy as fp
from dataclasses import dataclass
from types import MethodType

import nlpnet
import spacy
import stanza


@dataclass
class Token:
    text: str
    lemma: str
    pos_tag: str = None
    feats: str = None


def spacy_adaptor(doc):
    mapping = {'text': 'text',
               'lemma': 'lemma_',
               'pos_tag': 'pos_'
               }

    return [Token(**{key: getattr(token, value)
                     for key, value in mapping.items()})
            for token in doc
            ]


def stanza_adaptor(doc):
    mapping = {'text': 'text',
               'lemma': 'lemma',
               'pos_tag': 'upos',
               'feats': 'feats'
               }

    sentence = fp.first(doc.sentences)

    return [Token(**{key: getattr(token, value)
                     for key, value in mapping.items()})
            for token in sentence.words
            ]


def nlpnet_adaptor(doc):
    return [Token(**{'text': token,
                     'lemma': token,
                     'pos_tag': pos_tag})
            for token, pos_tag in fp.first(doc)
            ]


def get_adaptor_fn(nlp_pipeline):
    adaptor_fn = None
    for class_, adaptor in NLP_PIPELINE_MAP.items():
        if isinstance(nlp_pipeline, class_):
            adaptor_fn = adaptor
        elif isinstance(nlp_pipeline, MethodType):
            if isinstance(nlp_pipeline.__self__, class_):
                adaptor_fn = adaptor
    return adaptor_fn


NLP_PIPELINE_MAP = {
    stanza.pipeline.core.Pipeline: stanza_adaptor,
    spacy.language.Language: spacy_adaptor,
    nlpnet.taggers.POSTagger: nlpnet_adaptor
}


