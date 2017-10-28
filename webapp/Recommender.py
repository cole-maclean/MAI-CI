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
        self.graph=None
        self.session=None

    def load_graph(self):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile("model/frozen_model.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the 
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def, 
                input_map=None, 
                return_elements=None, 
                name="prefix", 
                op_dict=None, 
                producer_op_list=None
            )
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(graph=graph)
        self.input_tensor = graph.get_tensor_by_name('prefix/InputData/X:0')
        self.output_tensor = graph.get_tensor_by_name("prefix/FullyConnected/Softmax:0")
        return graph

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
        client_secret = config.get('reddit', 'client_secret')
        password = config.get('reddit','password')
        username = config.get('reddit', 'username')

        r = praw.Reddit(user_agent=reddit_user_agent,client_id = client_id,
                        client_secret=client_secret, username=username,
                        password=password) #initialize the praw Reddit object
        praw_user = r.redditor(user)
        user_data = [(user_comment.subreddit.display_name,
                      user_comment.created_utc) for user_comment in praw_user.comments.new(limit=None)]
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
        if self.graph == None:
            print("loading model")
            self.model = self.load_graph()
        sub_probs = self.session.run(self.output_tensor, feed_dict={
            self.input_tensor: X
        })
        filtered_probs = [[prob if i not in user_sub_seq else 0 for i,prob in enumerate(prob_list)] for prob_list in sub_probs]
        recs = [np.argmax(probs) for probs in filtered_probs]
        if recs:
            top_x_recs,cnt = zip(*Counter(recs).most_common(n_recs))
            sub_recs = [self.labels[sub_index] for sub_index in top_x_recs]
        else:
            sub_recs = []
        return sub_recs






        

