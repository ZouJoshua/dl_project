#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/10/19 11:36 PM
@File    : ner_with_StanfordNLP.py
@Desc    : 

"""


import re
from nltk.tag import StanfordNERTagger
import os
import pandas as pd
import nltk

def parse_document(document):
   document = re.sub('\n', ' ', document)
   if isinstance(document, str):
       document = document
   else:
       raise ValueError('Document is not string!')
   document = document.strip()
   sentences = nltk.sent_tokenize(document)
   sentences = [sentence.strip() for sentence in sentences]
   return sentences

# sample document
text = """
FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, 
Denmark, France, Germany, the Netherlands, Spain, Sweden, and Switzerland. Headquartered in Zürich, its 
membership now comprises 211 national associations. Member countries must each also be members of one of 
the six regional confederations into which the world is divided: Africa, Asia, Europe, North & Central America 
and the Caribbean, Oceania, and South America.
"""

sentences = parse_document(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# set java path in environment variables
java_path = r'C:\Program Files\Java\jdk1.8.0_161\bin\java.exe'
os.environ['JAVAHOME'] = java_path
# load stanford NER
sn = StanfordNERTagger('E://stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz',
                       path_to_jar='E://stanford-ner-2018-10-16/stanford-ner.jar')

# tag sentences
ne_annotated_sentences = [sn.tag(sent) for sent in tokenized_sentences]
# extract named entities
named_entities = []
for sentence in ne_annotated_sentences:
   temp_entity_name = ''
   temp_named_entity = None
   for term, tag in sentence:
       # get terms with NE tags
       if tag != 'O':
           temp_entity_name = ' '.join([temp_entity_name, term]).strip() #get NE name
           temp_named_entity = (temp_entity_name, tag) # get NE and its category
       else:
           if temp_named_entity:
               named_entities.append(temp_named_entity)
               temp_entity_name = ''
               temp_named_entity = None

# get unique named entities
named_entities = list(set(named_entities))
# store named entities in a data frame
entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
# display results
print(entity_frame)