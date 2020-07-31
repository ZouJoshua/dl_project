#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:38 PM
@File    : s_polyglot.py
@Desc    : 

"""

from polyglot.text import Text

from feature_extractor.multilanguage_keywords_extractor.core import base
from feature_extractor.multilanguage_keywords_extractor.model.sentence.s_base import Sentence
from feature_extractor.multilanguage_keywords_extractor.util import file_utils

class SentencePolyglot(Sentence):
    def __init__(self):
        self.__supportLanguages = set(file_utils.readLanguages(base.languagePath))

    def sentence(self, text, language):
        assert language in self.__supportLanguages, 'SentencePolyglot do not support {} language.'.format(language)
        sentences = Text(text).sentences
        return [sentence.raw for sentence in sentences]

    def supportLanguages(self):
        return self.__supportLanguages