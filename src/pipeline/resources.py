import json
import pandas as pd
from nltk.corpus import stopwords
from functools import lru_cache
from .. import settings

POLARITIES = ['Neg', None, 'Pos']
SENTILEX_POLARITIES = ['-1', '0', '1']
EMOTICON_POLARITIES = ['-1', '1']
INTERNAL_TOKENS = {'URL', 'HASH_TAG', 'USERNAME', 'NUMBER'}
SELECTED_POS_TAGS = ['ADJ', 'ADV', 'NOUN', 'VERB']
POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB']


@lru_cache()
def load_corpus():
    with open(settings.CORPUS_PATH, 'r') as file:
        json_content = file.read()

    parsed_json_content = json.loads(json_content)

    return pd.DataFrame([value for key, value in parsed_json_content.items()])


@lru_cache()
def load_sentilex():
    with open(settings.SENTILEX_PATH, 'r') as file:
        content = file.read()

    return dict(line.split(',') for line in content.strip().split('\n'))


@lru_cache()
def load_emoji_polarity_index():
    emoji_sentiment_frame = pd.read_csv(settings.EMOJI_SENTIMENT_PATH,
                                        usecols=['#Emoji', 'Occurrences', 'Negative', 'Neutral', 'Positive'])
    emoji_sentiment_frame = emoji_sentiment_frame.loc[lambda f: f['Occurrences'] > 0]
    emoji_sentiment_frame['polarity_index'] = emoji_sentiment_frame[
        ['Negative', 'Neutral', 'Positive']].to_numpy().argmax(axis=1)

    return {item['#Emoji']: item['polarity_index'] for _, item in
            emoji_sentiment_frame[['#Emoji', 'polarity_index']].iterrows()}


@lru_cache()
def load_stopwords():
    return set(stopwords.words('portuguese')) - set(['não'])
