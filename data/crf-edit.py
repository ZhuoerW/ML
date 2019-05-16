#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:27:17 2019

@author: WZE
"""

import re
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite



f_train = open("train-twi.txt","r")
f_test = open("test-twi.txt","r")

train_raw = f_train.read()
test_raw = f_test.read()
train_raw = train_raw.split("\n\n")
test_raw = test_raw.split("\n\n")
print(len(train_raw))
print(len(test_raw))

train_sents = []
for i in range(len(train_raw)):
	if len(train_raw[i]) > 0:
		train_words = train_raw[i].split("\n")
		train_split_word = []
		for j in range(len(train_words)):
			if len(train_words[j]) > 0:
				train_word = train_words[j].split(" ")
				train_split_word.append(train_word)
		train_sents.append(train_split_word)

test_sents = []
for i in range(len(test_raw)):
	if len(test_raw[i]) > 0:
		test_words = test_raw[i].split("\n")
		test_split_word = []
		for j in range(len(test_words)):
			if len(test_words[j]) > 0:
				test_word = test_words[j].split(" ")
				test_split_word.append(test_word)
		test_sents.append(test_split_word)

def word2feature(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    biotag = sent[i][2]
    features =[
		'bias',
        'word.lower=' + word.lower(),
		'word=' + word,
		'postag=' + postag,
		'biotag=' + biotag,
		'bio/pos=' + biotag + '/' + postag,
		'pos/word=' + postag + '/' + word,
		'bio/pos/word=' + biotag + '/' + postag + '/' + word,
        'word.isupper=%s' % word.isupper(),
        ]
    
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        biotag1 = sent[i - 1][2]
        features.extend([
              '-1:word.lower=' + word1.lower(),
              '-1:word.lower=%s' % word1.lower(),
              '-1:word.upper=%s' % word1.upper(),
			'-1:word=' + word1,
			'-1:postag=' + postag1,
			'-1:biotag=' + biotag1,
            ])
    else:
        features.append('BOS')
    return features
    if i>1:
        word1 = sent[i - 2][0]
        postag1 = sent[i - 2][1]
        biotag1 = sent[i - 2][2]
        features.extend([
              '-2:word.lower=' + word1.lower(),
              '-2:word.lower=%s' % word1.lower(),
              '-2:word.upper=%s' % word1.upper(),
			'-2:word=' + word1,
			'-2:postag=' + postag1,
			'-2:biotag=' + biotag1,
            ])
    
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        biotag1 = sent[i + 1][2]
        features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.lower=%s' % word1.lower(),
                '+1:word.upper=%s' % word1.upper(),
			'+1:word=' + word1,
			'+1:postag=' + postag1,
			'+1:biotag=' + biotag1,
        ])
    else:
        features.append('EOS')
    
    if i< len(sent)-2:
        word1 = sent[i + 2][0]
        postag1 = sent[i + 2][1]
        biotag1 = sent[i + 2][2]
        features.extend([
                '+2:word.lower=' + word1.lower(),
                '+2:word.lower=%s' % word1.lower(),
                '+2:word.upper=%s' % word1.upper(),
			'+2:word=' + word1,
			'+2:postag=' + postag1,
			'+2:biotag=' + biotag1,
        ])

def sent2features(sent):
    return [word2feature(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    label = []
    for i in range(len(sent)):
        label.append(sent[i][3])
    return label

def sent2tokens(sent):
	token = []
	for i in range(len(sent)):
		token.append(sent[i][0])
	return token

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

print("1")
trainer = pycrfsuite.Trainer(verbose=False)
print("2")
for xseq, yseq in zip(X_train, y_train):
	trainer.append(xseq, yseq)
print("3")
trainer.set_params({
        'c1': 0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 150,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})
print("4")
trainer.train('model')
print("5")
tagger = pycrfsuite.Tagger()
tagger.open('model')

#example_sent = test_sents[0]
#print(' '.join(sent2tokens(example_sent)), end='\n\n')

#print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
#print("Correct:  ", ' '.join(sent2labels(example_sent)))


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


y_pred = [tagger.tag(xseq) for xseq in X_test]

print(bio_classification_report(y_test, y_pred))

