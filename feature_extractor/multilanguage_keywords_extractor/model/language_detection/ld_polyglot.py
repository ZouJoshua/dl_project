#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:29 PM
@File    : ld_polyglot.py
@Desc    : 

"""

from polyglot.text import Text

from feature_extractor.multilanguage_keywords_extractor.core import base
from feature_extractor.multilanguage_keywords_extractor.model.language_detection.ld_base import LanguageDetection
from feature_extractor.multilanguage_keywords_extractor.core.language import unknownLanguage
from feature_extractor.multilanguage_keywords_extractor.util import file_utils

class LanguageDetectionPolyglot(LanguageDetection):
    def __init__(self):
        languages = file_utils.readLanguages(base.language_path)
        self.__support_languages = dict([(language.code, language) for language in languages])

    def detect(self, text):
        text = Text(text)
        code = text.language.code
        return self.__support_languages.get(code, unknownLanguage)

    def supportLanguages(self):
        return self.__support_languages