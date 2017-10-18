import numpy as np
import json
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
from tensorflow.python.framework import graph_util
     
        
class BuildRecommender():

    """

    """

    def __init__(self):
        self.unit_ids = self.load_unit_ids()
        self.vocab = self.load_vocab()
        self.max_seq_len = 88
        self.race_units = self.load_race_units()
        self.load_graph()
     
    def load_race_units(self):
        with open("race_units.json", 'r') as infile:
            race_units = json.load(infile)
        return race_units

    def load_vocab(self):
        with open("vocab.json", 'r') as  infile:
            vocab = json.load(infile)
        return vocab

    def load_unit_ids(self):
        with open("unit_ids.json", 'r') as infile:
            unit_ids = json.load(infile)
            unit_ids = {int(unit_id):name for unit_id,name in unit_ids.items()}
        return unit_ids
  
    def pred_preprocessing(self,pred_input):
        X = [[self.vocab.index(unit) for unit in pred_input]]
        X = pad_sequences(X, maxlen=self.max_seq_len, value=0.,padding='post')
        return X  
        
    def load_graph(self):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile("C:/Users/macle/Desktop/Open Source Projects/autocraft/webapp/model/model.pb", "rb") as f:
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
        self.graph = graph
        return graph
    
    def predict(self,pred_input):
        if not self.graph:
            self.freeze_graph()
            self.load_graph()
        pred_input = self.pred_preprocessing(pred_input)
        x = self.graph.get_tensor_by_name('prefix/InputData/X:0')
        y = self.graph.get_tensor_by_name("prefix/FullyConnected/Softmax:0")
        with tf.Session(graph=self.graph) as sess:
            build_probs = sess.run(y, feed_dict={
                x: pred_input
            })
        return build_probs
    
    def recurse_predictions(self,pred_input,preds,races,copy_vocab):
        rec_build = np.argmax(preds[0])
        rec = copy_vocab[rec_build]
        unit = rec[0:-1]
        player = int(rec[-1])
        if rec not in pred_input and unit in self.race_units[races[player][0:-1]]:
            return rec
        else:
            preds = np.delete(preds,rec_build)
            copy_vocab.pop(rec_build)
            return self.recurse_predictions(pred_input,preds,races,copy_vocab)       
    
    def predict_build(self,pred_input,build_length,races):
        #using list creates copy
        copy_vocab = list(self.vocab)
        rec_builds = []
        copy_input = list(pred_input)
        for i in range(build_length):
            build_probs = self.predict(copy_input)
            rec = self.recurse_predictions(copy_input,build_probs,races,copy_vocab)
            copy_input.append(rec)
            rec_builds.append(rec)
        return rec_builds