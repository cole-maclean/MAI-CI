#TODO: Error catching/ pipeling step checking
#TODO: add params to train_network

import rnn
import json
import pandas as pd
import numpy as np
from boltons.setutils import IndexedSet
from tflearn.data_utils import pad_sequences
from joblib import Parallel, delayed
from operator import itemgetter

np.random.seed(seed=42)

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def normalize(lst):
    s = sum(lst)
    normed = [itm/s for itm in lst]
    normed[-1] = (normed[-1] + (1-sum(normed)))#pad last value with what ever differenc eto make sum to 1
    return normed


class SubRecommender():

    def __init__(self, train_data_file='',embedding_file='',sequence_chunk_size = 51,min_seq_length=5, batch_size = 256,min_count_thresh=10):
        self.sequence_chunk_size = sequence_chunk_size
        self.train_data_file = train_data_file
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.training_sequences = []
        self.training_labels = []
        self.training_seq_lengths = []
        self.vocab = []
        self.min_count_thresh = min_count_thresh
        self.embedding_file = embedding_file

    def load_train_df(self,load_file = ''):
        print("Loading Training Data")
        if load_file:
            print ("Data loaded from disk")
            cache_df = pd.read_json(load_file)
            self.training_sequences = [[self.vocab.index(sub) for sub in seq if sub in self.vocab] for seq in cache_df['sub_seqs']]
            self.training_labels = [self.vocab.index(sub) for sub in cache_df['sub_label'] if sub in self.vocab]
            self.training_seq_lengths = cache_df['seq_length']
            return pd.DataFrame({'sub_seqs':self.training_sequences,'sub_label':self.training_labels,'seq_length':self.training_seq_lengths})
        with open(self.train_data_file,'r') as data_file:
            train_data = json.load(data_file)
        with open("data/user_comment_sequence_cache.json",'r') as cache_file:
            cache_data = json.load(cache_file)
        if self.vocab == []:
            self.create_vocab()
        df = pd.DataFrame(train_data,columns=['user','subreddit','utc_stamp'])
        df['utc_stamp'] = pd.to_datetime(df['utc_stamp'],unit='s')
        df.sort_values(by=['user','utc_stamp'], ascending=True, inplace=True)
        users = list(df.groupby('user')['user'].nunique().keys())
        self.training_sequences = []
        self.training_labels = []
        self.training_seq_lengths = []
        print("Building Training Sequences")
        for i,usr in enumerate(users):
            if usr in cache_data.keys():
                usr_sub_seq = cache_data[usr]
            else:
                user_comment_subs = list(df.loc[df['user'] == usr]['subreddit'].values)
                usr_sub_seq = [] #build sequence of non-repeating subreddit interactions
                for i,sub in enumerate(user_comment_subs):
                    if i ==0:
                        usr_sub_seq.append(sub)
                    elif sub != user_comment_subs[i-1]:#Check that current sub isn't repeated action of previous sub
                        usr_sub_seq.append(sub)
            cache_data[usr] = usr_sub_seq
        with open("data/user_comment_sequence_cache.json",'w') as cache_file:
            json.dump(cache_data,cache_file)
        rslts = Parallel(n_jobs=6)(delayed(self.build_training_sequences)(usr) for usr in cache_data.values())
        self.training_sequences = [data[0] for seq_chunks in rslts for data in seq_chunks]
        self.training_labels = [data[1] for seq_chunks in rslts for data in seq_chunks]
        self.training_seq_lengths = [data[2] for seq_chunks in rslts for data in seq_chunks]
        train_df = pd.DataFrame({'sub_seqs':self.training_sequences,'sub_label':self.training_labels,'seq_length':self.training_seq_lengths})
        cache_df = pd.DataFrame({'sub_seqs':[[self.vocab[vc_indx] for vc_indx in seq] for seq in self.training_sequences],
                                 'sub_label':[self.vocab[vc_indx] for vc_indx in self.training_labels],'seq_length':self.training_seq_lengths})
        cache_df.to_json("data/training_sequences/" + str(self.vocab_size) + "_" + str(self.sequence_chunk_size) + "_" + str(self.min_seq_length) + "_sequence_data.json")
        return train_df

    def build_training_sequences(self,usr_sub_seq):
        user_labels = []
        train_seqs = []
        comment_chunks = chunks(usr_sub_seq,self.sequence_chunk_size)
        print("vocab size = " + str(self.vocab_size))
        for chnk in comment_chunks:
            filtered_subs = [self.vocab.index(sub) for sub in chnk if sub in self.vocab and sub not in user_labels]
            if filtered_subs:
                filter_probs = normalize([self.vocab_probs[sub_indx] for sub_indx in filtered_subs])
                label = np.random.choice(filtered_subs,1,p=filter_probs)[0]
                user_labels.append(label)
                chnk_seq = [self.vocab.index(sub) for sub in chnk if sub in self.vocab and self.vocab.index(sub) != label] 
                if len(chnk_seq) > self.min_seq_length:
                    train_seqs.append([chnk_seq,label,len(chnk_seq)]) 
        return train_seqs

    def create_vocab(self):
        print("Building Vocab")
        with open(self.train_data_file,'r') as data_file:
            train_data = json.load(data_file)
        df = pd.DataFrame(train_data,columns=['user','subreddit','utc_stamp'])
        vocab_counts = df["subreddit"].value_counts()
        total_counts = len(df["subreddit"])
        with open(self.embedding_file,'r') as data_file:
            embeddings = json.load(data_file)
        tmp_vocab = [sub for sub,cnt in vocab_counts.items() if sub in embeddings.keys() and cnt >= self.min_count_thresh]
        inv_prob = [total_counts/vocab_counts[sub] for sub in tmp_vocab]
        self.embedding = [[0.0,0.0]] + [embeddings[sub] for sub in tmp_vocab]
        self.vocab = ["Unseen-Sub"] + tmp_vocab
        tmp_vocab_probs = normalize(inv_prob)
        self.vocab_probs = [1-sum(tmp_vocab_probs)] + tmp_vocab_probs #force probs sum to 1 by adding differenc to "Unseen-sub" probability
        self.vocab_size = len(self.vocab)
        self.idx_to_vocab = dict(enumerate(self.vocab))
        self.vocab_to_idx = dict(zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))
        return self.vocab

    def split_train_test(self,train_df,split_perc):
        train_len, test_len = np.floor(len(train_df)*split_perc), np.floor(len(train_df)*(1-split_perc))
        train, test = train_df.ix[:train_len-1], train_df.ix[train_len:train_len + test_len]
        return train, test 

    def train(self,load_file='',num_epochs=10):
        if self.vocab == []:
            self.create_vocab()
        if self.training_labels == []:
            train_df = self.load_train_df(load_file)
        else:
            train_df = pd.DataFrame({'sub_seqs':self.training_sequences,'sub_label':self.training_labels,'seq_length':self.training_seq_lengths})
        train,test = self.split_train_test(train_df,0.8)
        print("Training Model")
        self.model = rnn.train_model(train,test,self.vocab_size,self.embedding,self.sequence_chunk_size,num_epochs=num_epochs)
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