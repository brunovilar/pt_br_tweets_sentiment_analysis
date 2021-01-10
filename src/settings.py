import os

CORPUS_PATH = os.path.join(os.pardir, 'data', 'corpus', 'corpusTT.json')
SENTILEX_PATH = os.path.join(os.pardir, 'data', 'resources', 'sentilex-reduzido.txt')
EMOJI_SENTIMENT_PATH = os.path.join(os.pardir, 'data', 'resources', 'Emoji_Sentiment_Data_v1.0.csv')
NLPNET_POS_TAGGER_PATH = os.path.join(os.pardir, 'data', 'resources', 'pos-pt')
LOGS_ARTIFACTS_PATH = os.path.join(os.pardir, 'data', 'log')

NEGATIONS_WORDS = {'jamais','nada', 'nem','nenhum', 'nenhures', 'ninguém',
                   'ninguem', 'nonada', 'nulidade', 'nunca', 'não', 'nao',
                   'tampouco', 'zero'}

POSITIVE_EMOTICONS =  {':-)',':)',':o)',':]',':3',':c)',':>','=]','8)','=)',':}',':^)',
                       ':-))','|;-)',":'-)",":')",'\o/','*\\0/*',':-D',':D','8-D','8D',
                       'x-D','xD','X-D','XD','=-D','=D','=-3','=3','B^D','<3',';-)',';)',
                       '*-)','*)',';-]',';]',';D',';^)',':-,'}

NEGATIVE_EMOTICONS = {'>:\\','>:/',':-/',':-.',':/',':\\','=/','=\\',':L','=L',':S','>.<',
                      ':-|','<:-|','>:[',':-(',':(',':-c',':c',':-<',':<',':-[',':[',':{',
                      ':-||',':@',":'-(",":'(",'D:<','D:','D8','D;','D=','DX','v.v',"D-':",
                      '(>_<)',':|'}

