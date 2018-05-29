#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:27:43 2018

@author: sanzee
"""

import numpy as np
import pickle
class DimensionException(Exception):
    def __init__(self,reason,msg):
        self.expression = reason
        self.message = msg

class Glove_parser():
    def __init__(self,filename,vector_dim):
        self.filename = filename
        self.__vector_dim = vector_dim
        self.__word2vec = {}
        
        with open(filename,"r") as f:
            for line in f:
                curline = line.split()
                Word = curline[0]
                #print("getting vector for {}...".format(Word))
                vector = np.array(curline[1:]).astype(np.float)
                if len(vector) != vector_dim:
                    raise DimensionException("vector_dim != len(vector)","parsed dimension is different than the expected one")
                self.__word2vec[Word] = vector
            print("Done!!")
            print("Total words found: {}".format(len(self.__word2vec)))
            
                
    def get_dict(self):
        return self.__word2vec
    
    def save_glove(self,filename):
        print("Saving word2vector dict to {}".format(filename))
        with open(filename,"wb") as f:
            pickle.dump(self.__word2vec,f)
        print("Done!!")





