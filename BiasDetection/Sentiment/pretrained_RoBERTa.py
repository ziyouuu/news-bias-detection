"""
Using this model: https://huggingface.co/fhamborg/roberta-targeted-sentiment-classification-newsarticles
"""
import csv
import os
import sys
from os import get_terminal_size
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import nltk # Make sure this is downloaded -> nltk.download('punkt')
import pandas as pd
from itertools import chain

from NewsSentiment import TargetSentimentClassifier
tsc = TargetSentimentClassifier()

ABS_FILE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

INPUT_FILE = ABS_FILE_PATH + "../../datasets/news_dataset.csv"
TOKENIZER = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
NER_MODEL = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
NLP_PIPELINE = pipeline("ner", model=NER_MODEL, tokenizer=TOKENIZER, device=0)

'''
Load entire csv into memory
For each Document:
-> Split document into sentences using nltk
-> For each sentence:
-> -> Tokenise and NER sentences (choose first option)
-> Run Target Sentiment Classification on sentences
-> print count of sentiments in document
'''
def clear_lines(count=1):
    for n in range(1, count+1):
      sys.stdout.write("\033[F")  # Move up
      sys.stdout.write("\033[K")  # Clear lines
    sys.stdout.flush()

def clear_and_reset_terminal_line():
    columns, _ = get_terminal_size()
    print('', end='\r')  # return cursor to beginning
    print(' ' * (columns - 1), end='\r')  # Fill line with spaces

def ner_sentences(document):
  list_NERSentences = []
  for sentence in nltk.sent_tokenize(document):
    ner_spans = NLP_PIPELINE(sentence)
    for span in ner_spans:
      left = sentence[:span['start']]
      named_entity = sentence[span['start']:span['end']]
      right = sentence[span['end']:]
      list_NERSentences.append((left,named_entity,right))
      break # Only take first sentence. I'm too lazy to figure out the datatype NLP_PIPELINE returns to do this properly. I hate python sm
  return list_NERSentences

def inferSentiments(list_NERSentences):
  sentiments = tsc.infer(targets=list_NERSentences)
  return sentiments

def sanity(lines):
  with open(INPUT_FILE, 'r') as f:
    f.readline() # Skip headers
    for i in range(lines):
      print(f"=== Document{i} ===")
      line = f.readline()
      c = csv.reader([line]).__next__();
      ner = ner_sentences(c[4])
      sn = inferSentiments(ner)
      for i, result in enumerate(sn):
        print("===\nSentence:\n", ner[i], "\nSentiment:\n", result[0]['class_label']) #pick the first result

def batch(output_file_name):
  output_file = open(ABS_FILE_PATH + "../../datasets/generated/" + output_file_name, "w", buffering=1)
  output_file.write("unique_id,negative,neutral,positive\n")
  with open(INPUT_FILE, 'r') as f:
    f.readline() # Skip headers
    idx = 0
    while line := f.readline():
      print(f'Processing Document Number: {idx}')
      line = f.readline()
      c = csv.reader([line]).__next__();
      ner = ner_sentences(c[4])
      sn = inferSentiments(ner)
      dict_sentiments = {'negative': 0, 'neutral': 0, 'positive': 0}
      for i, result in enumerate(sn):
        dict_sentiments[result[0]['class_label']]+= 1
      clear_lines(2)
      idx += 1
      output_file.write(c[0]+ "," + str(dict_sentiments['negative'])+ "," + str(dict_sentiments['neutral'])+ "," + str(dict_sentiments['positive']) + "\n")
  output_file.close()

def ner_sentence_a(docs):
  list_NERSentences = []
  list_num_of_sents_in_doc = []
  for doc in docs:
    sents = list(doc.sents)
    sentsCount = 0
    for s in sents:
      if (len(s.ents) == 0):
        continue
      text = s.text
      ne = list(s.ents)[0].text
      ner_idx = text.index(ne)
      left = text[:ner_idx]
      named_entity = ne
      right = text[ner_idx + len(ne):]
      list_NERSentences.append((left,named_entity,right))
      sentsCount += 1
    list_num_of_sents_in_doc.append(sentsCount)
  return (list_NERSentences, list_num_of_sents_in_doc)

CHUNK_SIZE = 100
def batch_psu_safe():
  nlp = spacy.load("en_core_web_sm")
  output_file = open(ABS_FILE_PATH + "../../datasets/generated/news_dataset_sentiment_labels.csv", "w", buffering=1)
  output_file.write("unique_id,negative,neutral,positive\n")
  result = { 'unique_id': [], 'ner_sentences': [], 'sentiments': []}
  chunkIndependentDocumentId = 0
  for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, usecols=['unique_id', 'article_text']):
    unique_ids = chunk['unique_id']
    sentiments = []
    print("=== CSV Imported. Starting Tokenizing ===")
    articles = chunk['article_text'].tolist()
    docs = tqdm(nlp.pipe(articles,n_process=8), total=len(articles))

    print("=== Tokenizing Completed. Starting NER ===")
    list_NERSentences, list_num_of_sents_in_doc = ner_sentence_a(docs)
    print("=== NER Complete. Starting Inference ===")
    sentiments = inferSentiments(list_NERSentences)
    dict_sentiments = {'negative': 0, 'neutral': 0, 'positive': 0}
    documentIdx = 0
    sentencesIdx = 0
    for i, result in enumerate(sentiments):
      if (list_num_of_sents_in_doc[documentIdx] == sentencesIdx):
        output_file.write(f'{unique_ids[chunkIndependentDocumentId]},{dict_sentiments["negative"]},{dict_sentiments["neutral"]},{dict_sentiments["positive"]}\n')
        dict_sentiments = {'negative': 0, 'neutral': 0, 'positive': 0}
        documentIdx += 1
        chunkIndependentDocumentId +=1
        sentencesIdx = 0
      dict_sentiments[result[0]['class_label']]+= 1
      sentencesIdx+=1
    output_file.write(f'{unique_ids[chunkIndependentDocumentId]},{dict_sentiments["negative"]},{dict_sentiments["neutral"]},{dict_sentiments["positive"]}\n')
    chunkIndependentDocumentId +=1
    clear_lines(6)
  output_file.close()

def main():
  val = input("Pick mode:\n1. Sanity Check\n2. Create dump\n3. Create dump (psu safe)\n") 
  if (val == '1'):
    val = int(input("Enter number of documents to parse:"))
    sanity(val)
    return
  if (val == '2'):
    val = input("Enter output file name:")
    batch(val)
    return
  if (val == '3'):
    batch_psu_safe()
    return

if __name__=="__main__":
    main()
