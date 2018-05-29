#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:15:54 2018

@author: Muhammad Rifayat Samee (sanzee)

Model :
    Embedding ---> BILSTM ---->Softmax
    pretrained embedding (glove)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Create_vocab import Vocab,DataSet
from Glove_parsing import Glove_parser

### Hyper Parameters ###

USE_CUDA = False
BATCH_SIZE = 10
learning_rate = 0.001
dropout = 0.5
embedding_dim = 100 # word embedding dimension
lstm_layer = 2
Epoch = 5
grad_clip = 5.0
char_embed_dim = 50 # charecter embedding dimension
dropout_lstm = 0.55
dropout_char = 0
### end ####
###
#Glove_parser("glove.6B/glove.6B.100d.txt",100).save_glove("globe100d.pkl")
#word2vec dict is save as globe100d.pkl (if not run the previous line)

###

vocab = Vocab("data/all.txt")
vocab.add_new_word('<pad>')
vocab.add_new_label('<pad>')
pad_val = vocab._word2index['<pad>']
pad_lebel = vocab._label2index['<pad>']

train_data = DataSet("data/train.txt",vocab._word2index,vocab._label2index,vocab._char2index).get_data()
test_data = DataSet("data/test.txt",vocab._word2index,vocab._label2index,vocab._char2index).get_data()

def next_batch(total_sample,iteration,batch_size=100):
    start_idx = iteration*batch_size
    end_idx = min(total_sample,start_idx + batch_size)
    cur_batch = train_data[start_idx:end_idx]
    np.random.shuffle(cur_batch)
    input_seq = []
    tag_seq = []
    for i in range(len(cur_batch)):
        input_seq.append(cur_batch[i][2])
        tag_seq.append(cur_batch[i][3])
        
    return input_seq, tag_seq

def test_evalution(batch_size=10):
    Model.train(False)
    num_batch = len(test_data) // batch_size
    y_pred = []
    y_true = []
    for i in range(num_batch):
        
        start_idx = i*batch_size
        end_idx = min(len(test_data),start_idx + batch_size)
        input_seq = []
        tag_seq = []
        for j in range(start_idx,end_idx):
            input_seq.append(test_data[j][2])
            tag_seq.append(test_data[j][3])
        logits,target,_ = Model(input_seq,tag_seq)
        y_pred = y_pred + list(torch.argmax(logits.view(-1,logits.size(2)),1).cpu().numpy())
        y_true = y_true + list(target.view(-1).cpu().numpy())
        if i%100 ==0:
            print("evalution done for batch {} of total {}".format(i,num_batch))
    total = 0
    corrected = 0
    for i in range(len(y_true)):
        if y_true[i] != pad_val:
            total +=1
            if y_true[i]  == y_pred[i]:
                corrected +=1
    print("evaluated completed !! Acc : {}",corrected/total)
    Model.train(True)
    return

def create_pretain_word2vec_matrix(dim):
    words = list(vocab._word2index.keys())
    with open("globe100d.pkl","rb") as f:
        word2vec = pickle.load(f)
    pre_train_matrix = torch.zeros((len(words),dim))
    for w in words:
        if w.lower() not in word2vec.keys():
            pre_train_matrix[vocab._word2index[w]] = torch.tensor(np.random.normal(size=(dim,)))
        else:
            pre_train_matrix[vocab._word2index[w]] = torch.tensor(word2vec[w.lower()])
    return pre_train_matrix

    
class BILSTM_SOFTMAX(nn.Module):
    def __init__(self,vocab_size,num_class,hidden_dimension=300,embedding_dim=100,char_embedding_dim = char_embed_dim):
        super(BILSTM_SOFTMAX, self).__init__()
        W_Matrix = create_pretain_word2vec_matrix(embedding_dim)
        self.word_embed = nn.Embedding(vocab_size,embedding_dim=embedding_dim)
        self.word_embed.load_state_dict({'weight':W_Matrix})
        self.bilstm = nn.LSTM(embedding_dim,hidden_dimension,num_layers=lstm_layer,bidirectional=True,dropout=dropout_lstm)
        self.bilstm_char = nn.LSTM(char_embedding_dim,hidden_dimension,num_layers=lstm_layer,bidirectional=True,dropout=dropout_char)
        self.fc = nn.Linear(2*hidden_dimension,num_class)
    
    def forward(self,input_seq,target):
        #print(len(input_seq))
        #print(len(target))
        real_seq_len = torch.LongTensor([len(x) for x in input_seq])
        padded_seq = torch.zeros([len(input_seq),real_seq_len.max()]).long()
        padded_seq = padded_seq.new_full(padded_seq.size(),vocab._word2index['<pad>'])
        padded_tar = torch.zeros([len(target),real_seq_len.max()]).long()
        padded_tar = padded_tar.new_full(padded_tar.size(),vocab._label2index['<pad>'])
        #print(padded_seq.size(),padded_tar.size(),real_seq_len.size())
        for i in range(len(input_seq)):
            padded_seq[i,:real_seq_len[i]] = torch.LongTensor(input_seq[i])
            #print(i,real_seq_len[i],len(target[i]))
            padded_tar[i,:real_seq_len[i]] = torch.LongTensor(target[i])
        #soring them for pad_pack
        real_seq_len,sorting_permu = real_seq_len.sort(0,descending=True)
        padded_seq = padded_seq[sorting_permu]
        padded_tar = padded_tar[sorting_permu]
        #making length,batch,tensor
        padded_seq = padded_seq.transpose(0,1)
        embedded = self.word_embed(padded_seq)
        #print("embedded : ",embedded.size())
        packed_embedded = pack_padded_sequence(embedded,real_seq_len)
        #print("packed embedded : ",embedded.size())
        packed_output,(hidden_state,cell_state) = self.bilstm(packed_embedded)
        output,_ = pad_packed_sequence(packed_output)
        output = output.transpose(0,1)
        output = self.fc(output)
        logits = F.log_softmax(output,dim=2)
        #print("output",logits.size())
               
        
        return logits, padded_tar, real_seq_len
    
Model = BILSTM_SOFTMAX(len(vocab._word2index),len(vocab._label2index))
optimizer = optim.SGD(Model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-5)
LossFunc = nn.CrossEntropyLoss()
#batch_seq,batch_tar = next_batch(len(train_data),0,batch_size=10)
#print(len(batch_seq[0]),len(batch_tar[0]))
#logits,padded_tar,real_seq_len = Model(batch_seq,batch_tar)


def train():
    num_batch = len(train_data) // BATCH_SIZE
    for e in range(Epoch):
        print("Running Epoch {} of total {}".format(e,Epoch))
        
        for no_bacth in range(num_batch):
        
            batch_seq,batch_tar = next_batch(len(train_data),no_bacth,batch_size=BATCH_SIZE)
            #print(len(batch_seq[0]),len(batch_tar[0]))
            
            Model.train(True)
            Model.zero_grad()
            logits,padded_tar,_ = Model(batch_seq,batch_tar)
            loss = LossFunc(logits.view(-1,logits.size(2)),padded_tar.view(-1))
            print("running batch number {}/{} current loss {}".format(no_bacth,num_batch,loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm(Model.parameters(),grad_clip)
            optimizer.step()
        
        test_evalution()
        np.random.shuffle(train_data)
    
if __name__ == '__main__':
    train()

            
        
        
        
        
        
        
        
        
        
        




        
        
        
        
        
        
        
        
        
    