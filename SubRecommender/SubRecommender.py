#TODO: Error catching/ pipeling step checking
#TODO: add params to train_network

import rnn
import json
import pandas as pd
import random
import numpy as np
from boltons.setutils import IndexedSet
from tflearn.data_utils import pad_sequences

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

class SubRecommender():

    def __init__(self, sequence_chunk_size = 51,min_seq_length=5, train_data_file='',batch_size = 256,save_model_file=''):
        self.sequence_chunk_size = sequence_chunk_size
        self.train_data_file = train_data_file
        self.save_model_file = save_model_file
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.training_sequences = []
        self.training_labels = []
        self.training_seq_lengths = []

    def load_train_df(self):
        print("Loading Training Data")
        with open(self.train_data_file,'r') as data_file:
            train_data = json.load(data_file)
        with open("data/user_comment_sequence_cache.json",'r') as cache_file:
            sequence_cache = json.load(cache_file)
        df = pd.DataFrame(train_data,columns=['user','subreddit','utc_stamp'])
        df['utc_stamp'] = pd.to_datetime(df['utc_stamp'],unit='s')
        df.sort_values(by=['user','utc_stamp'], ascending=True, inplace=True)
        users = list(df.groupby('user')['user'].nunique().keys())
        sub_list = self.create_vocab(list(df.groupby('subreddit')['subreddit'].nunique().keys()))
        self.training_sequences = []
        self.training_labels = []
        self.training_seq_lengths = []
        print("Building Training Sequences")
        for i,usr in enumerate(users):
            if i % int(len(users)/10) == 0:
                print("Sequence Builder " + str(round(i/len(users)*100,0)) + " % Complete")
            if usr in sequence_cache.keys():
                usr_sub_seq = sequence_cache[usr]
            else:
                user_comment_subs = list(df.loc[df['user'] == usr]['subreddit'].values)
                usr_sub_seq = [] #build sequence of non-repeating subreddit interactions
                for i,sub in enumerate(user_comment_subs):
                    if i ==0:
                        usr_sub_seq.append(sub)
                    elif sub != user_comment_subs[i-1]:#Check that current sub isn't repeated action of previous sub
                        usr_sub_seq.append(sub)
                sequence_cache[usr] = usr_sub_seq
            comment_chunks = chunks(usr_sub_seq,self.sequence_chunk_size)
            for chnk in comment_chunks:
                label = sub_list.index(IndexedSet(chnk)[-1])#Last interacted with subreddit in chunk
                chnk_seq = [sub_list.index(sub) for sub in chnk if sub_list.index(sub) != label] 
                if len(chnk_seq) > self.min_seq_length:
                    self.training_sequences.append(chnk_seq)  
                    self.training_seq_lengths.append(len(chnk_seq))
                    self.training_labels.append(label)
        with open("data/user_comment_sequence_cache.json",'w') as cache_file:
            json.dump(sequence_cache,cache_file)
        return pd.DataFrame({'sub_seqs':self.training_sequences,'sub_label':self.training_labels,'seq_length':self.training_seq_lengths})

    def create_vocab(self,sub_reddits):
        print("Building Vocab")
        self.vocab = ["Unseen-Sub"] + sub_reddits
        self.vocab_size = len(self.vocab)
        self.idx_to_vocab = dict(enumerate(self.vocab))
        self.vocab_to_idx = dict(zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))
        return self.vocab

    def split_train_test(self,train_df,split_perc):
        train_len, test_len = np.floor(len(train_df)*split_perc), np.floor(len(train_df)*(1-split_perc))
        self.train, self.test = train_df.ix[:train_len-1], train_df.ix[train_len:train_len + test_len]
        return self.train, self.test 

    def train(self):
        if self.training_labels == []:
            train_df = self.load_train_df()
        train,test = self.split_train_test(train_df,0.8)
        self.model = rnn.train_model(train,test,self.vocab_size,self.sequence_chunk_size)
        return self.model

    def recommend_subs(self,user_data):
        rec_subs = []
        for usr in user_data:
            usr_sub_seq = [] #build sequence of non-repeating subreddit interactions
            for i,sub in enumerate(usr):
                if i ==0:
                    usr_sub_seq.append(sub)
                elif sub != usr[i-1]:#Check that current sub isn't repeated action of previous sub
                    usr_sub_seq.append(sub)
            usr_seqs = [[self.vocab_to_idx[sub] if sub in self.vocab else 0 for sub in chnk] for chnk in chunks(usr_sub_seq,self.sequence_chunk_size)]
            pad_usr_seqs = pad_sequences(usr_seqs, maxlen=self.sequence_chunk_size, value=0.)
            sub_probs = self.model.predict(pad_usr_seqs)
            rec_subs.append([self.idx_to_vocab[probs.index(max(probs))] for probs in sub_probs])
        return rec_subs

