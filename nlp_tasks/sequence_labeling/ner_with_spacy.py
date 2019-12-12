#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/12/19 4:24 PM
@File    : ner_with_spacy.py
@Desc    : 

"""


import spacy
# python -m spacy download en
#import pattern
import nltk

s = "In 2006, a trial court had convicted Salman Khan and gave him jail sentences of one and five years respectively for allegedly " \
    "poaching three chinkaras on two separate occasions.Jodhpur: The Rajasthan High Court, on Monday, acquitted Salman Khan in two " \
    "cases related to alleged poaching of Chinkaras.  In 2006, a trial court had convicted Salman Khan and gave him jail sentences " \
    "of one and five years respectively for allegedly poaching three chinkaras on two separate occasions. However, the Rajasthan High " \
    "Court exonerated Salman Khan on the following grounds: 1) The High Court found that the evidence on record is inadequate 2) The " \
    "FIR was registered on the basis of the statement of Harish Dulani, driver of Salman Khan's gypsy. However the witness was not " \
    "examined or cross-examined during the trial 3) The HC said it is doubtful whether Harish Dulani was a witness to the incidents. 4) The place " \
    "where the incidents allegedly occurred was never identified. 5) No carcass of Chinkara was found, so the prosecution could not " \
    "establish it was killed on September 26. 6) Pellets were not found on October 7 from the Gypsy used for hunting. But their sudden " \
    "discovery on October 12 gave rise to suspicion. 7) The court said it is impossible to kill a deer with an air gun. 8) The recovered " \
    "pellets are used for hunting small animals like rabbit or birds and not big animals. "

nlp = spacy.load("en")
doc = nlp(s)
sentence = list()
noun_chunks = list()
noun_phrase = list()
for ent in doc.ents:
 #   print(ent.text, ent.label_)
     if ent.label_ != 'DATE' and ent.label_ != 'TIME' and ent.label_ != 'LANGUAGE' and ent.label_ != 'MONEY' and ent.label_ != 'QUANTITY' and ent.label_ != 'ORDINAL' and ent.label_ != 'CARDINAL':
        #if len(ent.text.split(" ")) >= 2:
            sentence.append(ent.text.lower())
print(list(set(sentence)))
for nc in doc.noun_chunks:
    if len(nc.text.split(" ")) >= 2:
        sentence.append(nc.text.lower())
print(list(set(sentence)))


# res = nltk.parse(s)
# print(res)