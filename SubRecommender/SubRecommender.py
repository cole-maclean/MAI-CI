#TODO: Error catching/ pipeling step checking
#TODO: add params to train_network

import rnn
import json
import pandas as pd
import random
import numpy as np

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def build_user_comment_df(sub_cmt_list,chunk_size,batch_size):
    comment_chunks = chunks(sub_cmt_list,chunk_size)
    usr_seqs = [chnk for chnk in comment_chunks]
    padded_seqs = pad_pred_data(usr_seqs,batch_size)
    df = pd.DataFrame({'sub_seqs':padded_seqs})
    df['sub_label'] = -1
    df['seq_length'] = df.apply (lambda row: len(row['sub_seqs']),axis=1)
    return df

def pad_pred_data(pred_data,batch_size):
    i = len(pred_data)
    while i < batch_size:
        pred_data.append([0])
        i = i + 1
    return pred_data

class SubRecommender():

    def __init__(self, sequence_chunk_size = 25, train_data_file='',batch_size = 256,save_model_file=''):
        self.sequence_chunk_size = sequence_chunk_size
        self.train_data_file = train_data_file
        self.save_model_file = save_model_file
        self.batch_size = batch_size

    def load_train_df(self):
        with open(self.train_data_file,'r') as data_file:
            train_data = json.load(data_file)
        df = pd.DataFrame(train_data,columns=['user','subreddit','utc_stamp'])
        df['utc_stamp'] = pd.to_datetime(df['utc_stamp'],unit='s')
        df.sort_values(by=['user','utc_stamp'], ascending=True, inplace=True)
        users = list(df.groupby('user')['user'].nunique().keys())
        sub_list = self.create_vocab(list(df.groupby('subreddit')['subreddit'].nunique().keys()))
        self.training_sequences = []
        self.training_labels = []
        self.training_seq_lengths = []
        for usr in users:
            user_comment_subs = list(df.loc[df['user'] == usr]['subreddit'].values)
            comment_chunks = chunks(user_comment_subs,self.sequence_chunk_size)
            for chnk in comment_chunks:
                label = sub_list.index(random.choice(chnk))
                self.training_labels.append(label)
                chnk_seq = [sub_list.index(sub) for sub in chnk if sub_list.index(sub) != label]
                self.training_sequences.append(chnk_seq)  
                self.training_seq_lengths.append(len(chnk_seq))
        return pd.DataFrame({'sub_seqs':self.training_sequences,'sub_label':self.training_labels,'seq_length':self.training_seq_lengths})

    def create_vocab(self,sub_reddits):
        self.vocab = sub_reddits
        self.vocab_size = len(self.vocab)
        self.idx_to_vocab = dict(enumerate(self.vocab))
        self.vocab_to_idx = dict(zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))
        return self.vocab

    def split_train_test(self,train_df,split_perc):
        train_len, test_len = np.floor(len(train_df)*split_perc), np.floor(len(train_df)*(1-split_perc))
        self.train, self.test = train_df.ix[:train_len-1], train_df.ix[train_len:train_len + test_len]
        return self.train, self.test 

    def train_network(self):
        train_df = self.load_train_df()
        train,test = self.split_train_test(train_df,0.8)
        self.g = rnn.build_graph(vocab=self.vocab,batch_size=self.batch_size)
        tr_losses, te_losses = rnn.train_graph(self.g,train,test,num_epochs=1,batch_size=self.batch_size,
                                   save=self.save_model_file)
        return tr_losses,te_losses

    def rec_subs(self,sub_cmt_list):
        user_df = build_user_comment_df(sub_cmt_list,self.sequence_chunk_size,self.batch_size)
        return rnn.recommend_subs(self.g,self.save_model_file,user_df,self.batch_size)