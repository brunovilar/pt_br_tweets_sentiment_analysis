import regex as re
from .regular_expressions import (IDENTIFY_HTML_TAGS, IDENTIFY_URLS, IDENTIFY_REPEATED_SPACES,
                                  IDENTIFY_HASH_TAGS, IDENTIFY_MENTIONS, IDENTIFY_NUMBERS,
                                  IDENTIFY_REPETITIONS, MODIFY_REPETITIONS)


def clean_text(text, unify_html_tags=True, unify_urls=True, trim_repeating_spaces=True, unify_hashtags=True,
               unify_mentions=True, unify_numbers=True, trim_repeating_letters=True):
    text = re.sub(IDENTIFY_HTML_TAGS, ' ', text) if unify_html_tags else text
    text = re.sub(IDENTIFY_URLS, 'URL', text) if unify_urls else text
    text = re.sub(IDENTIFY_REPEATED_SPACES, ' ', text) if trim_repeating_spaces else text
    text = re.sub(IDENTIFY_HASH_TAGS, ' HASH_TAG ', text) if unify_hashtags else text
    text = re.sub(IDENTIFY_MENTIONS, ' USERNAME ', text) if unify_mentions else text
    text = re.sub(IDENTIFY_NUMBERS, ' NUMBER ', text) if unify_numbers else text
    text = re.sub(IDENTIFY_REPETITIONS, MODIFY_REPETITIONS, text) if trim_repeating_letters else text

    return text
