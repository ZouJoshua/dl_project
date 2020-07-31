#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/5/19 1:47 PM
@File    : file_utils.py
@Desc    : 

"""


from feature_extractor.multilanguage_keywords_extractor.core.language import Language


def writeLines(filePath, lines):
    writer = open(filePath, 'w', encoding='utf-8')
    for line in lines:
        writer.write(line + '\n')
    writer.close()

def readLines(filePath):
    reader = open(filePath, 'r', encoding='utf-8')
    return [line.strip() for line in reader.readlines()]

def readLanguages(filePath):
    return [Language(kv[0], kv[1]) for kv in [line.split('\t') for line in readLines(filePath)]]

def readStopwords(filePath):
    return set(readLines(filePath))