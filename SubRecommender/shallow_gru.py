from __future__ import division, print_function, absolute_import


import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import dask.array as da
import numpy as np

def train_model(train,test,vocab_size,max_seq_size,chunks=1024,num_epochs=10,learning_rate=0.001,n_units=256,dropout=0.5):

    trainX = train['sub_seqs']
    trainY = train['sub_label']
    testX =  test['sub_seqs']
    testY =  test['sub_label']

    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=max_seq_size, value=0.,padding='post')
    testX = pad_sequences(testX, maxlen=max_seq_size, value=0.,padding='post')

    trainX = da.from_array(np.asarray(trainX), chunks=chunks)
    trainY = da.from_array(np.asarray(trainY), chunks=chunks)
    testX = da.from_array(np.asarray(testX), chunks=chunks)
    testY = da.from_array(np.asarray(testY), chunks=chunks)

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=vocab_size)
    testY = to_categorical(testY, nb_classes=vocab_size)

    # Network building
    net = tflearn.input_data([None, max_seq_size])
    net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128,trainable=True)
    net = tflearn.gru(net, n_units=n_units, dropout=dropout,weights_init=tflearn.initializations.xavier(),return_seq=False)
    net = tflearn.fully_connected(net, vocab_size, activation='softmax',weights_init=tflearn.initializations.xavier())
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net,tensorboard_dir='/tmp/tflearn_logs/shallow_gru/', tensorboard_verbose=2)
                        #checkpoint_path='/tmp/tflearn_logs/shallow_lstm/',
                        #best_checkpoint_path="C:/Users/macle/Desktop/UPC Masters/Semester 2/CI/SubRecommender/models/")

    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=False,snapshot_epoch=True,
              batch_size=256,n_epoch=num_epochs,run_id=str(learning_rate)+"-"+str(n_units)+"-"+str(dropout))
    
    return model