#!/usr/bin/env python
import numpy as np 
import scipy 
import scipy.sparse as ss 
import csv
import math
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from collections import Counter

char_list = "?,!.()'\":"

def preprocess( str_in ):
  s = str_in.lower()
  for c in char_list:
    if c in s:
      s = s.replace(c, ' ')
  return s.replace('-', '')

def get_word_dict( sentence_list ):
  word_dict = {}
  for sentence in sentence_list: # fills word_dict
    for word in sentence.split():
      if word not in word_dict:
        word_dict[word] = 1
      else:
        word_dict[word] += 1

  return word_dict

def get_pairs_dict( sentence_list , L ):
  pairs = Counter()
  for sentence in sentence_list: # do this for each sentence in sentence_list
    s = sentence.split()
    w_index = 0
    for word in s: # do this for each word in the sentence
      c_index = w_index - L 
      for _ in range(2*L): # maximum of 2*L wordpairs for each word
        if(c_index == w_index):
          c_index += 1
        if(c_index < 0):
          c_index += 1
          continue
        if(c_index > len(s)-1): # if len(s) is 9, the last index is s[8]. cant index greater than len(s)-1
          c_index += 1
          continue
        c = s[c_index]
        wordpair = (word, c)
        pairs[wordpair] += 1
        c_index += 1 
      w_index += 1 # go onto the next word, index++
  
  return dict(pairs)

def get_ppmi_matrix( pairs_dict , word_dict , key2index, D , n ):
  rows = np.arange(D)
  cols = np.arange(D)
  data = np.arange(D)

  i = 0
  for key, val in pairs_dict.items():
    rows[i] = key2index[key[0]]
    cols[i] = key2index[key[1]]
    pmi_val = math.log((val * D) / (word_dict[key[0]] * word_dict[key[1]]))
    ppmi_val = max(0, pmi_val)
    data[i] = ppmi_val
    i += 1

  ppmi_matrix_in_csr_format = csr_matrix((data, (rows, cols)), shape = (n, n))
  return ppmi_matrix_in_csr_format

def get_word_embedding_matrix( ppmi_matrix ):
  ppmi_matrixf64 = ppmi_matrix.asfptype()
  vals, vecs = scipy.sparse.linalg.eigs(ppmi_matrixf64, k = 100)
  S_k = np.diag(vals)
  V_k = vecs
  F = V_k @ S_k # 97502 x 100 ; word-embedding matrix ; each row is a k-dimensional embedding feature vector for a word
  return F

def compute_accuracy( sentence_list, answer_list, thr , F , key2index ):

  correct_pairs = 0
  total_pairs = len(answer_list)

  for i in range(len(answer_list)):
    q1 = sentence_list[2*i].split()
    q2 = sentence_list[2*i+1].split()

    fvec1 = F[0]*0
    for word in q1:
      fvec1 += F[key2index[word]]

    fvec2 = F[0]*0
    for word in q2:
      fvec2 += F[key2index[word]]

    fvec1 = (1 / max(1, len(q1))) * fvec1 # to prevent division by 0
    fvec2 = (1 / max(1, len(q2))) * fvec2

    cos_similarity = (fvec1 @ fvec2) / (np.linalg.norm(fvec1) * np.linalg.norm(fvec2))

    if(cos_similarity - thr <= 0):
      prediction = 0
    else:
      prediction = 1

    if(prediction == int(answer_list[i])):
      correct_pairs += 1
  accuracy = correct_pairs / total_pairs
  return accuracy # accuracy for the whole data set

def main():
  sentence_list = []
  answer_list = []

  filename = input("Filename: ")
  f = open(filename, 'r')
  reader = csv.reader(f)
  datalist = list(reader)
  f.close()

  for item in datalist: 
    q1 = preprocess(item[3])
    q2 = preprocess(item[4])
    sentence_list.append(q1)
    sentence_list.append(q2)
    answer_list.append(item[5]) # this is a 0 or 1 ; 0 = questions unrelated, 1 = questions related
  
  word_dict = get_word_dict(sentence_list)

  L = 3 # window size to get pairings from
  pairs_dict = get_pairs_dict(sentence_list, L)

  n = len(word_dict) # n is the number of distinct words in the dataset; 97502
  D = len(pairs_dict) # D is the number of distinct wordpairs in the dataset; 4918033

  # create a mapping of unique words to unique numbers, from {0, ... , n}
  keys = word_dict.keys()
  key2index = {}
  index2key = {}
  for index, key in enumerate(keys):
    key2index[key] = index
    index2key[index] = key  # indexes are from 0 to 97501

  ppmi_matrix = get_ppmi_matrix(pairs_dict, word_dict, key2index, D, n)
  
  F = get_word_embedding_matrix(ppmi_matrix)

  '''thr = 0.80
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 0.82
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 0.84
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 0.86
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 0.88
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 0.90
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")'''
  thr = 0.92
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  '''thr = 0.94
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 0.96
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 0.98
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")
  thr = 1.0
  accuracy = compute_accuracy(sentence_list, answer_list, thr, F, key2index)
  print("Threshold: ", thr, "Accuracy: ", accuracy, "\n")'''

if __name__ == '__main__':
  main()