#TODO: Error catching/ pipeling step checking
#TODO: add params to train_network

import RNN
import json
import pandas as pd
import random

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


class SubRecommender():

    def __init__(self, train_data_file='',batch_size = 256, test_data_file='',saved_model=''):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.saved_model = saved_model
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
            comment_chunks = chunks(user_comment_subs,25)
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

    def train_network(self,save=''):
        train_df = self.load_train_df()
        train,test = self.split_train_test(train_df,0.8)
        g = build_graph(vocab=self.vocab,batch_size=self.batch_size)
        tr_losses, te_losses = train_graph(g,train,test,num_epochs=1,batch_size=self.batch_size,
                                   save=save)
        return tr_losses,te_losses


    def generate_predictions(self, g, checkpoint, num_pred_subs,state=None, prompt_sequence=['AskReddit'], pick_top_subs=None):
        """ Accepts a current character, initial state"""

        with tf.Session() as sess:
            print(sess.run(tf.global_variables_initializer()))
            g['saver'].restore(sess, checkpoint)
            
            subs = []
            for seed_sub in prompt_sequence:
                current_sub = vocab_to_idx[seed_sub]
                subs.append(idx_to_vocab[current_sub])
                if state is not None:
                    feed_dict={g['x']: [[current_sub]], g['init_state']: state}
                else:
                    feed_dict={g['x']: [[current_sub]]}

            for i in range(num_pred_subs):
                if state is not None:
                    feed_dict={g['x']: [[current_sub]], g['init_state']: state}
                else:
                    feed_dict={g['x']: [[current_sub]]}

                preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

                if pick_top_subs is not None:
                    p = np.squeeze(preds)
                    p[np.argsort(p)[:-pick_top_subs]] = 0
                    p = p / np.sum(p)
                    current_sub = np.random.choice(vocab_size, 1, p=p)[0]
                else:
                    current_sub = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]
                subs.append(idx_to_vocab[current_sub])
        return subs