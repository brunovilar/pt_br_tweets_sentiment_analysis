import regex as re

IDENTIFY_HTML_TAGS = re.compile(r'<[^>]*>', re.IGNORECASE)
IDENTIFY_URLS = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+))', re.IGNORECASE)
IDENTIFY_REPEATED_SPACES = re.compile(r'\s{2, }', re.IGNORECASE)
IDENTIFY_HASH_TAGS = re.compile(r'(^|\s)([#][\w_-]+)', re.IGNORECASE)
IDENTIFY_MENTIONS = re.compile(r'(^|\s)([@][\w_-]+)', re.IGNORECASE)
IDENTIFY_NUMBERS = re.compile(r'([$]?[0-9]+,*[0-9]*)+', re.IGNORECASE)

IDENTIFY_REPETITIONS = r'(.)\1+'
MODIFY_REPETITIONS = r'\1\1'

EXTRACT_POLARITY = re.compile(r'Polarity=(\w*)([|]|$)', re.IGNORECASE)
