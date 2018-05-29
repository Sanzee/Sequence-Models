#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:49:28 2018

@author: Muhammad Rifayat Samee(sanzee)
"""
import os
import sys
import numpy as np
from copy import deepcopy
'''
Assuming that data files are tab (\t) separated and one scentence and tag per line

'''

class Vocab():
    def __init__(self,filename,sep='\t'):
        print("Preparing Vocab...")
        self.filename = filename
        self._word2index = {}
        self._index2word = {}
        self._label2index = {}
        self._index2label = {}
        self._char2index = {}
        self._index2char = {}
        self._total_word_count = 0
        self._total_label_count = 0
        self._total_char_count = 0
        self._Dataset = []
        with open(filename,"r") as f:
        
            Data = f.readlines()
            for i in range(len(Data)):
                sten,POS = Data[i].strip().split(sep)
                self._Dataset.append([sten.split(),POS.split()])
                for x in sten.split():
                    for char in x :
                        if char not in self._char2index:
                            self._char2index[char] = self._total_char_count
                            self._index2char[self._total_char_count] = char
                            self._total_char_count +=1
                    if x not in self._word2index:
                        self._word2index[x] = self._total_word_count
                        self._index2word[self._total_word_count] = x
                        self._total_word_count +=1
                        
                for x in POS.split():
                    if x not in self._label2index:
                        self._label2index[x] = self._total_label_count
                        self._index2label[self._total_label_count] = x
                        self._total_label_count +=1

        print("Total Token: ",len(self._word2index))
        print("Total char Token: ",len(self._char2index))
        print("Total Class: ",len(self._label2index))
        

    def add_new_word(self,word):
        if word in self._word2index:
            print("Word already exits!!")
            return
        
        self._word2index[word] = self._total_word_count
        self._index2word[self._total_word_count] = word
        self._total_word_count +=1
        for char in word:
            if char not in self._char2index:
                self._char2index[char] = self._total_char_count
                self._index2char[self._total_char_count] = char
                self._total_char_count +=1
        print("{} sucessfully added to the vocab as a word and also char dict( if needed) has been updated!!".format(word))
        return
    
    def add_new_label(self,label):
        if label in self._label2index:
            print("Label already exits!!")
            return
        self._label2index[label] = self._total_label_count
        self._index2label[self._total_label_count] = label
        self._total_label_count +=1
        print("{} sucessfully added to the vocab as tag!!".format(label))
        return
    

class DataSet():
    def __init__(self,filename,word2index,label2index,char2index,sep='\t'):
        print("Preparing DataSet....{}".format(filename))
        self._data = []
        
        with open(filename,"r") as f:
            Data = f.readlines()
            for i in range(len(Data)):
                sten,POS = Data[i].strip().split(sep)
                if len(sten.split()) != len(POS.split()):
                    print("Possible Length mismatch in the dataset!!")
                    exit()
                sten = sten.split()
                POS = POS.split()
                sten_indexed = []
                POS_indexed = []
                char_level = []
                char_index_level = []
                for j in range(len(sten)):
                    sten_indexed.append(word2index[sten[j]])
                    POS_indexed.append(label2index[POS[j]])
                    cur_word_char_index = []
                    cur_word_char = []
                    for char in sten[j]:
                        cur_word_char_index.append(char2index[char])
                        cur_word_char.append(char)
                    char_level.append(cur_word_char)
                    char_index_level.append(cur_word_char_index)
                    
                
                self._data.append([sten,POS,sten_indexed,POS_indexed,char_level,char_index_level])
                # char info
                
    
        print("Done.")
    def get_data(self):
        print("this function will return a deep copy version of the data processed!! it may takes time... ... please wait")
        return deepcopy(self._data)



if __name__ == "__main__":
    V = Vocab("all.txt")
    train_data = DataSet("train.txt",V._word2index,V._label2index,V._char2index).get_data()
    test_data = DataSet("test.txt",V._word2index,V._label2index,V._char2index).get_data()


    
