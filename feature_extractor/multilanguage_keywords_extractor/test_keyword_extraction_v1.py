#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:59 PM
@File    : test_keyword_extraction_v1.py
@Desc    : 

"""


import os

from feature_extractor.multilanguage_keywords_extractor.core import base
from feature_extractor.multilanguage_keywords_extractor.model.keyword_extraction.rake import \
    KeywordExtractionRake
from feature_extractor.multilanguage_keywords_extractor.model.keyword_extraction.textacy_singlerank import \
    KeywordExtractionTextacySinglerank
from feature_extractor.multilanguage_keywords_extractor.model.keyword_extraction.textacy_textrank import KeywordExtractionTextacyTextrank
from feature_extractor.multilanguage_keywords_extractor.model.keyword_extraction.tfidf import \
    KeywordExtractionTFIDF
from feature_extractor.multilanguage_keywords_extractor.model.language_detection.ld_polyglot import \
    LanguageDetectionPolyglot
from feature_extractor.multilanguage_keywords_extractor.model.stem.stem_nltk import StemNLTK
from feature_extractor.multilanguage_keywords_extractor.model.tokenization.token_polyglot import TokenizationPolyglot
from feature_extractor.multilanguage_keywords_extractor.util import file_utils

articlePath = base.data_path + 'zh/articles/'
keywordPath = base.data_path + 'zh/keywords/'

texts = [(fileName[0:fileName.find('.')], '.'.join(file_utils.readLines(articlePath + fileName))) for fileName in os.listdir(articlePath)]
print('texts:', texts)
oriKeywords = [(fileName[0:fileName.find('.')], ', '.join(file_utils.readLines(keywordPath + fileName))) for fileName in os.listdir(articlePath)]
print('oriKeywords:', oriKeywords)

languageDetection = LanguageDetectionPolyglot()
language = languageDetection.detect('\n'.join([text[1] for text in texts]))
print('language:', language)

stopwords = file_utils.readStopwords(base.punctuationStopwordPath)
tokenization = TokenizationPolyglot(stopwords)
stem = StemNLTK()

docs = [(text[0], stem.stem(tokenization.token(text[1], language), language)) for text in texts]

keywordExtractionTFIDF = KeywordExtractionTFIDF()
keywordExtractionTFIDF.train([doc[1] for doc in docs])

keywordExtractionTextacyTextrank = KeywordExtractionTextacyTextrank()
keywordExtractionTextacySinglerank = KeywordExtractionTextacySinglerank()
keywordExtractionRake = KeywordExtractionRake()
for i in range(len(docs)):
    print('OriKeywords:', oriKeywords[i][0], oriKeywords[i][1])
    print('KeywordExtractionTFIDF:', docs[i][0], ', '.join([keyword[0] for keyword in keywordExtractionTFIDF.extract(docs[i][1], 10)]))
    print('KeywordExtractionTextacyTextrank:', texts[i][0], ', '.join([keyword[0] for keyword in keywordExtractionTextacyTextrank.extract(texts[i][1], 10)]))
    print('KeywordExtractionTextacySinglerank:', texts[i][0], ', '.join([keyword[0] for keyword in keywordExtractionTextacySinglerank.extract(texts[i][1], 10)]))
    print('KeywordExtractionRake:', texts[i][0], ', '.join([keyword[0] for keyword in keywordExtractionRake.extract(texts[i][1], 10)]))
    print()