#TODO: Error catching/ pipeling step checking
#TODO: add params to train_network

import RNN
import json
import pandas as pd


class SubRecommender():

    def __init__(self, train_data_file='',test_data_file='',saved_model=''):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.saved_model = saved_model

    def load_train_sequence(self):
        with open(self.train_data_file,'r') as data_file:    
            train_data = json.load(data_file)
        df = pd.DataFrame(train_data,columns=['user','subreddit','utc_stamp'])
        df['utc_stamp'] = pd.to_datetime(df['utc_stamp'],unit='s')
        df.sort_values(by=['user','utc_stamp'], ascending=True, inplace=True)
        self.train_sub_seqs = []
        current_sub = ''
        for rw in df.iterrows():
            sub = rw[1]['subreddit']
            if sub != current_sub:
                self.train_sub_seqs.append(sub)   
            current_sub = sub
        return self.train_sub_seqs

    def create_vocab(self,train_sequence):
        self.vocab = set(train_sequence)
        self.vocab_size = len(vocab)
        self.idx_to_vocab = dict(enumerate(vocab))
        self.vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
        return self.vocab

    def train_network(self):
        train_seqs = self.load_train_sequence()
        self.create_vocab(train_seqs)
        data = [self.vocab_to_idx[c] for c in train_seqs]
        g = RNN.build_graph(num_classes = len(vocab),cell_type='LN_LSTM', num_steps=80)
        t = time.time()
        losses = train_network(g, data, 10, num_steps=80, save="models/LN_LSTM_10_epochs")
        print("It took", time.time() - t, "seconds to train for 10 epochs.")
        print("The average loss on the final epoch was:", losses[-1])

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