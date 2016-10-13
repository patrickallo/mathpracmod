"""
Module for textprocessing functions
"""

import re
import nltk

def tokenize(text, notext_pattern=re.compile("[^a-zA-Z0-9]"):
	"""
    takes unicode-text and returns tokenized content
    """
	filtered_text = notext_pattern.sub(" ", text)
    tokens = [word.lower() for sent in nltk.sent_tokenize(filtered_text)
              for word in nltk.word_tokenize(sent)]
    return tokens

def stem(tokens):
	"""
	takes tokens and returns stems using SnowballStemmer
	"""
	stemmer = nltk.stem.snowball.SnowballStemmer("english")
    stems = [stemmer.stem(token) for token in tokens]
    return stems

def tokenize_and_stem(text, notext_pattern=re.compile("[^a-zA-Z0-9]")):
    """
    takes unicode-text and returns tokenized and stemmed
    and just tokenized content
    """
    tokens = tokenize(text, notext_pattern)
    stems = stem(tokens)
    return tokens, stems


def strip_proppers_pos(text):
    """takes text as string as input and returns list of words that are
    not propper names"""
    tagged = nltk.tag.pos_tag(text.split())  # use NLTK's speech tagger
    non_propernouns = [word for word, pos in tagged if
                       pos != 'NNP' and pos != 'NNPS']
    return non_propernouns
