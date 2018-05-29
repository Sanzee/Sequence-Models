#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:35:39 2018

@author: bob
"""

import pickle
import numpy as np
from copy import deepcopy

def readfile(filename):
    with open(filename,"r") as f:
        data = f.readlines()
    
    
    return data

def check_indi_data(Data,filename=""):
        
    for i in range(len(Data)):
        Sent,POS = Data[i].split("\t")
        if len(Sent.split()) != len(POS.split()):
            print("something wrong in {}. encountered".format(filename))
            return -1
    
    print("{} is OK".format(filename))
    return 0

def check_with_sense(Data,Data_s,filename=""):
    if len(Data) != len(Data_s):
        print("something wrong in {}. encountered".format(filename))
        return -1
    
    for i in  range(len(Data)):
        S,POS = Data[i].split("\t")
        S_s,POS_s = Data_s[i].split("\t")
        if len(POS.split()) != len(S_s.split()):
            print("something wrong in {}. encountered".format(filename))
            return -1
    print("Cross check on {} is OK".format(filename))
    return 0


def Return_vocab(Data):
    vocab = set()
    POS_vocab = set()
    for i in range(len(Data)):
        S,POS = Data[i].split("\t")
        vocab = vocab.union(set(S.split()))
        POS_vocab = POS_vocab.union(set(POS.split()))
    print(len(vocab),len(POS_vocab))
    return vocab,POS_vocab

All_file = readfile("all.txt")
All_s_file = readfile("all_s.txt")
train_file = readfile("train.txt")
train_s_file = readfile("train_s.txt")
test_file = readfile("test.txt")
test_s_file = readfile("test_s.txt")

check_indi_data(All_file,"all.txt")
check_indi_data(All_s_file,"all_s.txt")
check_indi_data(train_file,"train.txt")
check_indi_data(train_s_file,"train_s.txt")
check_indi_data(test_file,"test.txt")
check_indi_data(test_s_file,"test_s.txt")

check_with_sense(All_file,All_s_file,filename="all.txt")
check_with_sense(train_file,train_s_file,filename="train.txt")
check_with_sense(test_file,test_s_file,filename="test.txt")

#Return_vocab(All_file) # 45050 45
#Return_vocab(All_s_file) # 46946 45

train_vocab,_ = Return_vocab(train_file)
test_vocab,_ = Return_vocab(test_file)

suffix  = readfile("suffix_.txt")

for i in range(len(suffix)):
    suffix[i] = suffix[i].strip()
    
train_suffix_list = {}
test_suffix_list = {}
i = 0
for suf in suffix:
    List_cur = []
    print(i)
    i = i + 1
    for word in train_vocab:
        if word.endswith(suf):
            List_cur.append(word)
    
    train_suffix_list[suf] = sorted(list(set(deepcopy(List_cur))),key=lambda x: len(x))
    
    List_cur = []
    for word in test_vocab:
        if word.endswith(suf):
            List_cur.append(word)
    
    test_suffix_list[suf] = sorted(list(set(deepcopy(List_cur))),key=lambda x: len(x))

import pickle
with open("train_suffix_list.pkl","rb") as f:
    a = pickle.load(f)
with open ("train_suffixes.txt", "w") as f:
    for key, value in a.items():
        #f.write("\n------suffix-----")
        f.write("-")
        f.write(key)
        f.write("\n")
        if len(value) == 0:
            f.write("NONE")
            f.write("\n")
            continue 
        for x in value:
            f.write(x)
            f.write("\n")
        f.write("\n")
        
with open("test_suffix_list.pkl","rb") as f:
    a = pickle.load(f)
with open ("test_suffixes.txt", "w") as f:
    for key, value in a.items():
        #f.write("\n------suffix-----")
        f.write("-")
        f.write(key)
        f.write("\n")
        if len(value) == 0:
            f.write("NONE")
            f.write("\n")
            continue 
        for x in value:
            f.write(x)
            f.write("\n")
        f.write("\n")
        