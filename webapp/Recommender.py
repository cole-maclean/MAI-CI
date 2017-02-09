import csv
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
from tflearn.layers.recurrent import bidirectional_rnn,GRUCell
import praw
import configparser
from collections import Counter



def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

class Recommender():

    def __init__(self):
        self.embedding_weights = np.load('model/lowDWeights.npy')
        self.labels = self.load_labels()
        self.model=None

    def load_model(self):
        net = tflearn.input_data([None, 15])
        net = tflearn.embedding(net, input_dim=5027, output_dim=128,trainable=True)
        net = tflearn.gru(net, n_units=128, dropout=0.6,weights_init=tflearn.initializations.xavier(),return_seq=False)
        net = tflearn.fully_connected(net, 5027, activation='softmax',weights_init=tflearn.initializations.xavier())
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.00093,
                                 loss='categorical_crossentropy')
        model = tflearn.DNN(net)
        model.load("model/shallow_gru.tfl",weights_only=True)
        return model

    def load_labels(self):
        labels = []
        with open('model/labels.tsv','r') as tsv:    
            for line in csv.reader(tsv):
                labels.append(line[0])
        return labels

    def collect_user_data(self,user):

        #Import configuration parameters, user agent for PRAW Reddit object
        config = configparser.ConfigParser()
        config.read('secrets.ini')

        #load user agent string
        reddit_user_agent = config.get('reddit', 'user_agent')
        client_id = config.get('reddit', 'client_id')
        client_secret = config.get('reddit', 'client_api_key')

        r = praw.Reddit(user_agent=reddit_user_agent,client_id = client_id,client_secret=client_secret) #initialize the praw Reddit object
        praw_user = r.get_redditor(user)
        user_data = [(user_comment.subreddit.display_name,
                      user_comment.created_utc) for user_comment in praw_user.get_comments(limit=None)]
        return sorted(user_data,key=lambda x: x[1]) #sort by ascending utc timestamp

    def user_recs(self,user,n_recs=10,chunk_size=15):
        user_data = self.collect_user_data(user)
        user_sub_seq = [self.labels.index(data[0]) if data[0] in self.labels else 0 for data in user_data]
        non_repeating_subs = []
        for i,sub in enumerate(user_sub_seq):
            if i == 0:
                non_repeating_subs.append(sub)
            elif sub != user_sub_seq[i-1]:
                non_repeating_subs.append(sub)
        self.user_subs = set([self.labels[sub_index] for sub_index in non_repeating_subs])
        sub_chunks = list(chunks(non_repeating_subs,chunk_size))
        X = pad_sequences(sub_chunks, maxlen=chunk_size, value=0.,padding='post')
        if self.model == None:
            print("loading model")
            self.model = self.load_model()
        sub_probs = self.model.predict(X)
        recs = [probs.index(max(probs)) for probs in sub_probs]
        filtered_recs = [filt_rec for filt_rec in recs if filt_rec not in user_sub_seq]
        top_x_recs,cnt = zip(*Counter(filtered_recs).most_common(n_recs))
        sub_recs = [self.labels[sub_index] for sub_index in top_x_recs]
        return sub_recs






        

