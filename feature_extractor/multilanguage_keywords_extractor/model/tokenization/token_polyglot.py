#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:42 PM
@File    : token_polyglot.py
@Desc    : 

"""

from polyglot.text import Text

from feature_extractor.multilanguage_keywords_extractor.core import base
from feature_extractor.multilanguage_keywords_extractor.model.tokenization.token_base import Tokenization
from feature_extractor.multilanguage_keywords_extractor.util import file_utils


class TokenizationPolyglot(Tokenization):
    def __init__(self, stopwords=set()):
        self.__support_languages = set(file_utils.readLanguages(base.language_path))
        self.__stopwords = stopwords

    def token(self, text, language):
        assert language in self.__support_languages, 'TokenizationPolyglot do not support {} language.'.format(language)
        words = Text(text).words
        return [word.lower() for word in words if word.lower() not in self.__stopwords]

    def supportLanguages(self):
        return self.__support_languages