#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/18/19 11:05 PM
@File    : ngram_with_tfidf.py
@Desc    : n-gram+TFIDF

"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer




x = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

#切割词袋
vectorizer = CountVectorizer(ngram_range=(1, 2))
# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
x = transformer.fit_transform(vectorizer.fit_transform(x))

print(vectorizer.get_feature_names())
print(x)
print(x.toarray())