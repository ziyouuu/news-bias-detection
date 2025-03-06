"""
Using this model: https://huggingface.co/fhamborg/roberta-targeted-sentiment-classification-newsarticles
"""
import csv
import os
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import nltk
# Make sure this is downloaded -> nltk.download('punkt')

from NewsSentiment import TargetSentimentClassifier
tsc = TargetSentimentClassifier()

INPUT_FILE = "../../datasets/news_dataset.csv"
LINES_TO_READ=1
TOKENIZER = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
NER_MODEL = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
NLP_PIPELINE = pipeline("ner", model=NER_MODEL, tokenizer=TOKENIZER)

'''
Load entire csv into memory
For each Document:
-> Split document into sentences using nltk
-> For each sentence:
-> -> Tokenise and NER sentences (choose first option)
-> Run Target Sentiment Classification on sentences
-> print count of sentiments in document
'''

def splitDoc2Sentences(document):
  return nltk.sent_tokenize(document)

def ner_sentences(sentences):
  list_NERSentences = []
  for sentence in sentences:
    ner_spans = NLP_PIPELINE(sentence)
    span = ner_spans[0] #get first NER span
    left = sentence[:span['start']]
    named_entity = sentence[span['start']:span['end']]
    right = sentence[span['end']:]
    list_NERSentences.append((left,named_entity,right))
  return list_NERSentences

def inferSentiments(list_NERSentences):
  sentiments = tsc.infer(targets=list_NERSentences)
  return sentiments

with open(INPUT_FILE, 'r') as f:
  f.readline() # Skip headers
  for i in range(LINES_TO_READ):
    line = f.readline()
    c = csv.reader([line])
    s = splitDoc2Sentences(c[4])
    ner = ner_sentences(s)
    sn = inferSentiments(ner)
    print(sn)
    for i, result in enumerate(sn):
      print("Sentiment:", i, result[0])