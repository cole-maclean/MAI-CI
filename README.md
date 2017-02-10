
<h2>Introduction</h2>
As part of a project course in my second semester, we were tasked with building a system of our chosing that encorporated or showcased any of the Computational Intelligence techniques we learned about in class. For our project, we decided to investigate the application of Recurrent Nueral Networks to the task of building a Subreddit recommender system for Reddit users. In this post, I outline some of the implementation details of the final system. A minimal webapp for the final model can be interacted with [here,](http://ponderinghydrogen.pythonanywhere.com/) The final research paper for the project can be found [here](http://cole-maclean.github.io/blog/files/subreddit-recommender.pdf) and my collaboraters on the project are Barbara Garza and Suren Oganesian. The github repo for the project can be found [here.](https://github.com/cole-maclean/MAI-CI)

![spez](documentation/images/spez.PNG)

<h3>Model Hypothesis</h3>

The goal of the project is to utilize the sequence prediction power of RNN's to predict possibly interesting subreddits to a user based on their comment history. The hypothesis of the recommender model is, given an ordered sequence of user subreddit interactions, patterns will emerge that favour the discovery of paticular new subreddits given that historical user interaction sequence. Intuitively speaking, as users interact with the Reddit ecosystem, they discover new subreddits of interest, but these new discoveries are influenced by the communities they have previously been interacting with. We can then train a model to recognize these emergent subreddit discoveries based on users historical subreddit discovery patterns. When the model is presented with a new sequence of user interaction, it "remembers" other users that historically had similiar interaction habits and recommends their subreddits that the current user has yet to discover.  
 
This sequential view of user interaction/subreddit discovery is similiar in structure to other problems being solved with the use of Recurrent Neural Networks, such as [Character Level Language Modelling](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Automatic Authorship Detection](http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html). Due to the successes of these similiarly structured problems, we have decided to explore RNN models for the subbreddit Recommendator System.

<h3>The Data</h3>
The secret sauce in any machine learning system, we need data. Reddit provides a convenient API for scrapping its public facing data, and the python package [PRAW](https://praw.readthedocs.io/en/latest/) is a popular and well documented wrapper that we used in this project. With the aim of developing sequences of user subreddit interactions, all we need for our raw data is a list of 3-tuples in the form [username,subreddit,utc timestamp]. The following script provides a helper function to collect and store random user comment data from Reddit's streaming 'all' comments. Note that PRAW authentication config data needs to be stored in a file named 'secret.ini' with:  
[reddit]  
api_key: key  
client_id: id  
client_api_key: client key  
redirect_url: redir url  
user_agent: subreddit-recommender by /u/upcmaici v 0.0.1  


```python
import praw
import configparser
import random
import pandas as pd
import numpy as np
import sys

#Import configuration parameters, user agent for PRAW Reddit object
config = configparser.ConfigParser()
config.read('secrets.ini')

#load user agent string
reddit_user_agent = config.get('reddit', 'user_agent')
client_id = config.get('reddit', 'client_id')
client_secret = config.get('reddit', 'client_api_key')

#main data scrapping script
def scrape_data(n_scrape_loops = 10):
    reddit_data = []
    #initialize the praw Reddit object
    r = praw.Reddit(user_agent=reddit_user_agent,client_id = client_id,client_secret=client_secret) 
    for scrape_loop in range(n_scrape_loops):
        try:
            all_comments = r.get_comments('all')
            print ("Scrape Loop " + str(scrape_loop))
            for cmt in all_comments:
                user = cmt.author        
                if user:
                    for user_comment in user.get_comments(limit=None):
                        reddit_data.append([user.name,user_comment.subreddit.display_name,
                                      user_comment.created_utc])
        except Exception as e:
            print(e)
    return reddit_data
```


```python
raw_data = scrape_data(10)
```

    Version 3.5.0 of praw is outdated. Version 4.3.0 was released Thursday January 19, 2017.
    Scrape Loop 0
    Scrape Loop 1
    Scrape Loop 2
    Scrape Loop 3
    Scrape Loop 4
    Scrape Loop 5
    Scrape Loop 6
    Scrape Loop 7
    Scrape Loop 8
    Scrape Loop 9
    


```python
print("Collected " + str(len(raw_data)) + " comments")
raw_data[0:10]
```

    Collected 158914 comments
    




    [['Illuminate1738', 'MapPorn', 1486680909.0],
     ['Illuminate1738', 'MapPorn', 1486471452.0],
     ['Illuminate1738', 'nova', 1486228887.0],
     ['Illuminate1738', 'nova', 1485554669.0],
     ['Illuminate1738', 'nova', 1485549461.0],
     ['Illuminate1738', 'MapPorn', 1485297397.0],
     ['Illuminate1738', 'ShitRedditSays', 1485261592.0],
     ['Illuminate1738', 'ShittyMapPorn', 1483836164.0],
     ['Illuminate1738', 'MapPorn', 1483798990.0],
     ['Illuminate1738', 'MapPorn', 1483503268.0]]



<h2>Data Munging</h2>
We need to parse the raw data into a structure consumpable by a supervised learning algorithm like RNN's. First we build a model vocabulary and ditribution of subreddit popularity from the collect raw data. We use this to build the training dataset, the subreddit interaction sequence for each user, ordered and then split into chunks representing different periods of Reddit interaction and discovery. From each chunk, we can randomly remove a single subreddit from the interaction as the "discovered" subreddit and use it as our training label for the interaction sequences. This formulation brings with it a hyperparameter that will require tuning, namely the sequence size of each chunk of user interaction periods. The proposed model utilizes the distribution of subreddits existing in the dataset to weight the random selection of a subreddit as the sequence label, which gives a higher probability of selection to rarer subreddits. This will smoothen the distribution of training labels across the models vocabulary of subreddits in the dataset. Also, each users interaction sequence has been compressed to only represent the sequence of non-repeating subreddits, to eliminate the repeatative structure of users constantly commenting in a single subreddit, while providing information of the users habits in the reddit ecosystem more generally, allowing the model to distinguish broader patterns from the compressed sequences.


```python
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def normalize(lst):
    s = sum(lst)
    normed = [itm/s for itm in lst]
    normed[-1] = (normed[-1] + (1-sum(normed)))#pad last value with what ever difference neeeded to make sum to exactly 1
    return normed
```


```python
"""This routine develops the models vocabulary and vocab_probs is also built, representing the inverse probability 
of encounting a paticular subreddit in the given dataset, which is then used to bias the selection of rarer
subreddits as labels to 
smooth the distribution of training labels across all subreddits in the vocabulary"""

df = pd.DataFrame(raw_data,columns=['user','subreddit','utc_stamp'])
train_data = None#free up train_data memory
vocab_counts = df["subreddit"].value_counts()
tmp_vocab = list(vocab_counts.keys())
total_counts = sum(vocab_counts.values)
inv_prob = [total_counts/vocab_counts[sub] for sub in tmp_vocab]
vocab = ["Unseen-Sub"] + tmp_vocab #build place holder, Unseen-Sub, for all subs not in vocab
tmp_vocab_probs = normalize(inv_prob)
#force probs sum to 1 by adding differenc to "Unseen-sub" probability
vocab_probs = [1-sum(tmp_vocab_probs)] + tmp_vocab_probs
print("Vocab size = " + str(len(vocab)))
```

    Vocab size = 3546
    


```python
sequence_chunk_size = 15
def remove_repeating_subs(raw_data):
    cache_data = {}
    prev_usr = None
    past_sub = None
    for comment_data in raw_data:
        current_usr = comment_data[0]
        if current_usr != prev_usr:#New user found in sorted comment data, begin sequence extraction for new user
            if prev_usr != None and prev_usr not in cache_data.keys():#dump sequences to cache for previous user if not in cache
                cache_data[prev_usr] = usr_sub_seq
            usr_sub_seq = [comment_data[1]] #initialize user sub sequence list with first sub for current user
            past_sub = comment_data[1]
        else:#if still iterating through the same user, add new sub to sequence if not a repeat
            if comment_data[1] != past_sub:#Check that next sub comment is not a repeat of the last interacted with sub,
                                            #filtering out repeated interactions
                usr_sub_seq.append(comment_data[1])
                past_sub = comment_data[1]
        prev_usr = current_usr #update previous user to being the current one before looping to next comment
    return cache_data

def build_training_sequences(usr_data):
    train_seqs = []
    #split user sub sequences into provided chunks of size sequence_chunk_size
    for usr,usr_sub_seq in usr_data.items():
        comment_chunks = chunks(usr_sub_seq,sequence_chunk_size)
        for chnk in comment_chunks:
            #for each chunk, filter out potential labels to select as training label, filter by the top subs filter list
            filtered_subs = [vocab.index(sub) for sub in chnk]
            if filtered_subs:
                #randomly select the label from filtered subs, using the vocab probability distribution to smooth out
                #representation of subreddit labels
                filter_probs = normalize([vocab_probs[sub_indx] for sub_indx in filtered_subs])
                label = np.random.choice(filtered_subs,1,p=filter_probs)[0]
                #build sequence by ensuring users sub exists in models vocabulary and filtering out the selected
                #label for this subreddit sequence
                chnk_seq = [vocab.index(sub) for sub in chnk if sub in vocab and vocab.index(sub) != label] 
                train_seqs.append([chnk_seq,label,len(chnk_seq)]) 
    return train_seqs
```

We transform the munged-data into a pandas dataframe for easier manipulation. Note that the subreddits have been integer encoded, indexed by their order in the vocabulary.


```python
pp_user_data = remove_repeating_subs(raw_data)
train_data = build_training_sequences(pp_user_data)
seqs,lbls,lngths = zip(*train_data)
train_df = pd.DataFrame({'sub_seqs':seqs,
                         'sub_label':lbls,
                         'seq_length':lngths})
train_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seq_length</th>
      <th>sub_label</th>
      <th>sub_seqs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>432</td>
      <td>[46, 157, 46, 483, 157, 46, 157, 856, 157, 856...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>46</td>
      <td>[157, 432, 157, 432, 157, 432, 157, 157, 157]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>46</td>
      <td>[432, 432, 432, 432, 856, 856, 157]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>432</td>
      <td>[46, 157, 46, 157, 46, 157, 856, 157, 46, 157,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>1048</td>
      <td>[46, 157, 46, 157, 46, 157, 46, 157, 46, 157, ...</td>
    </tr>
  </tbody>
</table>
</div>



<h3>Tensorflow Model Architecture</h3>

Originally, we built the model directly on-top of tensorflow, using the fantastic tutorials from [R2RT](http://r2rt.com/) as reference. However, building and managing various nueral network architectures with Tensorflow can be cumbersome, and higher level wrapper packages exist to abstract away some of the more tedious variable and graph definition steps required for tensorflow models. We chose the [tflearn](http://tflearn.org/) python package, which has an API similiar to sklearn, which the team had more experience with. With tflearn, it's rather easy to plug and play with different layers, and we experimented with LSTM, GRU and multi-layered Bi-Directional RNN architectures.


```python
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np

def train_model(train,test,vocab_size,n_epoch=2,n_units=128,dropout=0.6,learning_rate=0.0001):

    trainX = train['sub_seqs']
    trainY = train['sub_label']
    testX =  test['sub_seqs']
    testY =  test['sub_label']

    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=sequence_chunk_size, value=0.,padding='post')
    testX = pad_sequences(testX, maxlen=sequence_chunk_size, value=0.,padding='post')

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=vocab_size)
    testY = to_categorical(testY, nb_classes=vocab_size)

    # Network building
    net = tflearn.input_data([None, 15])
    net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128,trainable=True)
    net = tflearn.gru(net, n_units=n_units, dropout=dropout,weights_init=tflearn.initializations.xavier(),return_seq=False)
    net = tflearn.fully_connected(net, vocab_size, activation='softmax',weights_init=tflearn.initializations.xavier())
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=2)

    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=False,
              batch_size=256,n_epoch=n_epoch)
    
    return model
```

    c:\python35\lib\site-packages\tensorflow\python\util\deprecation.py:155: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
      arg_spec = inspect.getargspec(func)
    c:\python35\lib\site-packages\tensorflow\python\util\deprecation.py:155: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
      arg_spec = inspect.getargspec(func)
    c:\python35\lib\site-packages\tensorflow\contrib\labeled_tensor\python\ops\_typecheck.py:233: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
      spec = inspect.getargspec(f)
    

<h2>Model Training</h2>
We split the model into train/test sets and begin training. Here we use the default training parameters, but the model can be tuned for epochs, internal units, dropout, learning-rate and other hyperparameters of the chosen RNN structure.


```python
split_perc=0.8
train_len, test_len = np.floor(len(train_df)*split_perc), np.floor(len(train_df)*(1-split_perc))
train, test = train_df.ix[:train_len-1], train_df.ix[train_len:train_len + test_len]
model = train_model(train,test,len(vocab))
```

    Training Step: 29  | total loss: [1m[32m8.17396[0m[0m | time: 1.104s
    | Adam | epoch: 002 | loss: 8.17396 -- iter: 3584/3775
    Training Step: 30  | total loss: [1m[32m8.17391[0m[0m | time: 2.222s
    | Adam | epoch: 002 | loss: 8.17391 | val_loss: 8.17437 -- iter: 3775/3775
    --
    

It can be difficult to tell how well the model is performing simply by staring at the flipping numbers above, but tensorflow provides a visualization tool called [tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) and tflearn has different prebuilt dashboards which can be changed using the tensorboard_verbose option of the DNN layer.  

![tensorboard](documentation/images/tensorboard.PNG)

<h2>Visualizng the Model</h2>
As part of the model, a high dimension embedding space is learnt representing the subreddits in the vocabulary as vectors that can be reasoned about with "distance" from each other in the embedding space, and visualized with dimensionality reduction techniques, similiar to the concepts used in [word2vec.](http://www.deeplearningweekly.com/blog/demystifying-word2vec) The tutorial by Arthur Juliani [here](https://medium.com/@awjuliani/visualizing-deep-learning-with-t-sne-tutorial-and-video-e7c59ee4080c#.xdlzpd34w) was used to build the embedding visualization.


```python
from sklearn.manifold import TSNE
#retrieve the embedding layer fro mthe model by default name 'Embedding'
embedding = tflearn.get_layer_variables_by_name("Embedding")[0]
finalWs = model.get_weights(embedding)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
lowDWeights = tsne.fit_transform(finalWs)
```


```python
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.models import ColumnDataSource, LabelSet

#control the number of labelled subreddits to display
sparse_labels = [lbl if random.random() <=0.01 else '' for lbl in vocab]
source = ColumnDataSource({'x':lowDWeights[:,0],'y':lowDWeights[:,1],'labels':sparse_labels})


TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

p = figure(tools=TOOLS)

p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
          line_color=None,source=source)

labels = LabelSet(x="x", y="y", text="labels", y_offset=8,
                  text_font_size="10pt", text_color="#555555", text_align='center',
                 source=source)
p.add_layout(labels)

#output_file("embedding.html")
output_notebook()
show(p)
```

    c:\python35\lib\site-packages\bokeh\core\json_encoder.py:52: DeprecationWarning: parsing timezone aware datetimes is deprecated; this will raise an error in the future
      NP_EPOCH = np.datetime64('1970-01-01T00:00:00Z')
    



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="d2adfea4-106b-4ad7-afef-620b2772cb31">Loading BokehJS ...</span>
    </div>







    <div class="bk-root">
        <div class="bk-plotdiv" id="dd819e01-8269-4086-83ae-b38feebe6629"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("dd819e01-8269-4086-83ae-b38feebe6629").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("dd819e01-8269-4086-83ae-b38feebe6629");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'dd819e01-8269-4086-83ae-b38feebe6629' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"dbab9248-c350-4e62-963e-b3ebcfda68e9":{"roots":{"references":[{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"07160207-0b46-4ba3-b976-b9e49f5a2bd4","type":"ZoomOutTool"},{"attributes":{"dimension":1,"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"},"ticker":{"id":"b5238255-8855-4b02-ab6f-2c13f9393d00","type":"BasicTicker"}},"id":"a36cf62e-9bd1-4a97-bd69-eec7fa02ec56","type":"Grid"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"4b346730-b6ff-4f22-b8cb-cae3215ca111","type":"ResetTool"},{"attributes":{"overlay":{"id":"555b9bdd-ad50-472b-8711-8ffc2f0b4878","type":"BoxAnnotation"},"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"a0046ad7-f498-444b-a589-c4ebb0d7f042","type":"BoxZoomTool"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"b2c68d0f-5c52-4900-9097-dcebc5c9cea1","type":"BoxAnnotation"},{"attributes":{"callback":null,"overlay":{"id":"b2c68d0f-5c52-4900-9097-dcebc5c9cea1","type":"BoxAnnotation"},"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"},"renderers":[{"id":"0e91da5e-af49-4a0b-92e4-aac69573d67c","type":"GlyphRenderer"}]},"id":"d9cb1594-065f-4b23-96da-a5e77d734d8b","type":"BoxSelectTool"},{"attributes":{"callback":null,"overlay":{"id":"d7254aec-c5ba-4664-9c97-32b69636b6d9","type":"PolyAnnotation"},"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"883f5ef3-c254-48a9-b206-0deeb65ad248","type":"LassoSelectTool"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"8a1a9996-8574-4e77-94ec-97433eeccb23","type":"UndoTool"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"a0345ca8-000f-4526-b9f3-57aaef0a661b","type":"ZoomInTool"},{"attributes":{"fill_alpha":{"value":0.6},"fill_color":{"value":"#1f77b4"},"line_color":{"value":null},"radius":{"units":"data","value":0.1},"x":{"field":"x"},"y":{"field":"y"}},"id":"822b6453-268f-4624-a857-054cadb13f4b","type":"Circle"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"},"ticker":{"id":"832caed4-24d9-4bf1-b925-98c1655be318","type":"BasicTicker"}},"id":"073ca960-e1aa-40d0-86d7-d30a2b61d8fd","type":"Grid"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"577c045e-51e5-47b4-a3f0-01f619f37d29","type":"HoverTool"},{"id":"a174d10f-fa65-4a2e-809f-ea36b7c7a81b","type":"CrosshairTool"},{"id":"1aa993f5-2a64-45d0-8c59-e02e95c08dd6","type":"PanTool"},{"id":"1fe34764-900d-46d5-88cd-de9c16e13823","type":"WheelZoomTool"},{"id":"a0345ca8-000f-4526-b9f3-57aaef0a661b","type":"ZoomInTool"},{"id":"07160207-0b46-4ba3-b976-b9e49f5a2bd4","type":"ZoomOutTool"},{"id":"a0046ad7-f498-444b-a589-c4ebb0d7f042","type":"BoxZoomTool"},{"id":"8a1a9996-8574-4e77-94ec-97433eeccb23","type":"UndoTool"},{"id":"7e716b83-bf9a-4b5c-9f02-fd36825721b3","type":"RedoTool"},{"id":"4b346730-b6ff-4f22-b8cb-cae3215ca111","type":"ResetTool"},{"id":"9aec1b3e-08fd-4e31-a898-abb0cf7377d6","type":"TapTool"},{"id":"327690de-b6a9-401d-b2e0-20e7067e34cf","type":"SaveTool"},{"id":"d9cb1594-065f-4b23-96da-a5e77d734d8b","type":"BoxSelectTool"},{"id":"66d67d68-7614-49f6-a74c-3a3286b63655","type":"PolySelectTool"},{"id":"883f5ef3-c254-48a9-b206-0deeb65ad248","type":"LassoSelectTool"}]},"id":"1835c839-81d1-4c94-8db5-d79f89651d23","type":"Toolbar"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"a174d10f-fa65-4a2e-809f-ea36b7c7a81b","type":"CrosshairTool"},{"attributes":{},"id":"8c2a875b-194f-48bc-8c1b-6b111e7c1f56","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"xs_units":"screen","ys_units":"screen"},"id":"d7254aec-c5ba-4664-9c97-32b69636b6d9","type":"PolyAnnotation"},{"attributes":{"formatter":{"id":"2eb1e06f-cceb-4c3d-82cd-c5c34b78b40f","type":"BasicTickFormatter"},"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"},"ticker":{"id":"832caed4-24d9-4bf1-b925-98c1655be318","type":"BasicTicker"}},"id":"3b379f44-3f60-49c7-b4a8-51c5f2b183f9","type":"LinearAxis"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"radius":{"units":"data","value":0.1},"x":{"field":"x"},"y":{"field":"y"}},"id":"1ac0e2c1-7697-40f5-bc51-7a33754e63b8","type":"Circle"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"555b9bdd-ad50-472b-8711-8ffc2f0b4878","type":"BoxAnnotation"},{"attributes":{"plot":null,"text":""},"id":"78b2d3c6-f2a3-4d7b-9f6f-8b209a0cfc05","type":"Title"},{"attributes":{"callback":null,"column_names":["labels","x","y"],"data":{"labels":["","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","bestof","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","WorldofTanks","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","MorbidReality","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","TronMTG","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","gameswap","","","","","","4amShower","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Seattle","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Comcast","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","roadtrip","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","biggestproblem","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","oblivion","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","metalworking","Drag","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Floof","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","fightporn","","","","","","","","","","","","","","","","","","","","","","","","","utarlington","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Dank_Skeletons","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","matt01ss","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","betta","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","leangains","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Pikmin","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","finance","","","","","","","","","","","","droidturbo","","","","","","","","","","","","","","","","","","","","","","","","","","benzodiazepines","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Manitoba","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","YMS","","","","","","","","","","","","","","upvotegifs","","","","","","","","","","","","","","","","Incel","","","","","","","","HamRadio","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","HKdramas","","","","","GlitchInTheMatrix","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Vaping","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","onejob","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","fpvracing","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","SoylentMarket","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","arresteddevelopment","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Teleshits","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"x":{"__ndarray__":"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","dtype":"float64","shape":[3546]},"y":{"__ndarray__":"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","dtype":"float64","shape":[3546]}}},"id":"f38c0507-2455-4e95-9927-5ec366cb81d3","type":"ColumnDataSource"},{"attributes":{},"id":"832caed4-24d9-4bf1-b925-98c1655be318","type":"BasicTicker"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"327690de-b6a9-401d-b2e0-20e7067e34cf","type":"SaveTool"},{"attributes":{"callback":null,"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"9aec1b3e-08fd-4e31-a898-abb0cf7377d6","type":"TapTool"},{"attributes":{"below":[{"id":"3b379f44-3f60-49c7-b4a8-51c5f2b183f9","type":"LinearAxis"}],"left":[{"id":"18ed734f-aa0f-4950-b92c-2a5f03e58b22","type":"LinearAxis"}],"renderers":[{"id":"3b379f44-3f60-49c7-b4a8-51c5f2b183f9","type":"LinearAxis"},{"id":"073ca960-e1aa-40d0-86d7-d30a2b61d8fd","type":"Grid"},{"id":"18ed734f-aa0f-4950-b92c-2a5f03e58b22","type":"LinearAxis"},{"id":"a36cf62e-9bd1-4a97-bd69-eec7fa02ec56","type":"Grid"},{"id":"555b9bdd-ad50-472b-8711-8ffc2f0b4878","type":"BoxAnnotation"},{"id":"b2c68d0f-5c52-4900-9097-dcebc5c9cea1","type":"BoxAnnotation"},{"id":"ad51e956-a8b0-45e9-919a-29d03ffe1d4c","type":"PolyAnnotation"},{"id":"d7254aec-c5ba-4664-9c97-32b69636b6d9","type":"PolyAnnotation"},{"id":"0e91da5e-af49-4a0b-92e4-aac69573d67c","type":"GlyphRenderer"},{"id":"aa893c94-c487-4363-8b35-860be18fdd09","type":"LabelSet"}],"title":{"id":"78b2d3c6-f2a3-4d7b-9f6f-8b209a0cfc05","type":"Title"},"tool_events":{"id":"67755b45-15f7-4bf6-9ce1-94ba4bbe8965","type":"ToolEvents"},"toolbar":{"id":"1835c839-81d1-4c94-8db5-d79f89651d23","type":"Toolbar"},"x_range":{"id":"0176837e-f43c-4516-ba45-7bb6657e4aaa","type":"DataRange1d"},"y_range":{"id":"91e62546-99d9-4a1d-aa9e-d30cda2e0c5c","type":"DataRange1d"}},"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"67755b45-15f7-4bf6-9ce1-94ba4bbe8965","type":"ToolEvents"},{"attributes":{"callback":null,"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"577c045e-51e5-47b4-a3f0-01f619f37d29","type":"HoverTool"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"1aa993f5-2a64-45d0-8c59-e02e95c08dd6","type":"PanTool"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"xs_units":"screen","ys_units":"screen"},"id":"ad51e956-a8b0-45e9-919a-29d03ffe1d4c","type":"PolyAnnotation"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"1fe34764-900d-46d5-88cd-de9c16e13823","type":"WheelZoomTool"},{"attributes":{},"id":"b5238255-8855-4b02-ab6f-2c13f9393d00","type":"BasicTicker"},{"attributes":{"callback":null},"id":"0176837e-f43c-4516-ba45-7bb6657e4aaa","type":"DataRange1d"},{"attributes":{"callback":null},"id":"91e62546-99d9-4a1d-aa9e-d30cda2e0c5c","type":"DataRange1d"},{"attributes":{"overlay":{"id":"ad51e956-a8b0-45e9-919a-29d03ffe1d4c","type":"PolyAnnotation"},"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"66d67d68-7614-49f6-a74c-3a3286b63655","type":"PolySelectTool"},{"attributes":{"formatter":{"id":"8c2a875b-194f-48bc-8c1b-6b111e7c1f56","type":"BasicTickFormatter"},"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"},"ticker":{"id":"b5238255-8855-4b02-ab6f-2c13f9393d00","type":"BasicTicker"}},"id":"18ed734f-aa0f-4950-b92c-2a5f03e58b22","type":"LinearAxis"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"}},"id":"7e716b83-bf9a-4b5c-9f02-fd36825721b3","type":"RedoTool"},{"attributes":{"plot":{"id":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60","subtype":"Figure","type":"Plot"},"source":{"id":"f38c0507-2455-4e95-9927-5ec366cb81d3","type":"ColumnDataSource"},"text":{"field":"labels"},"text_align":"center","text_color":{"value":"#555555"},"text_font_size":{"value":"10pt"},"x":{"field":"x"},"y":{"field":"y"},"y_offset":{"value":8}},"id":"aa893c94-c487-4363-8b35-860be18fdd09","type":"LabelSet"},{"attributes":{"data_source":{"id":"f38c0507-2455-4e95-9927-5ec366cb81d3","type":"ColumnDataSource"},"glyph":{"id":"822b6453-268f-4624-a857-054cadb13f4b","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"1ac0e2c1-7697-40f5-bc51-7a33754e63b8","type":"Circle"},"selection_glyph":null},"id":"0e91da5e-af49-4a0b-92e4-aac69573d67c","type":"GlyphRenderer"},{"attributes":{},"id":"2eb1e06f-cceb-4c3d-82cd-c5c34b78b40f","type":"BasicTickFormatter"}],"root_ids":["84bed8a4-4b9c-41f2-8761-8b0ea94fde60"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"dbab9248-c350-4e62-963e-b3ebcfda68e9","elementid":"dd819e01-8269-4086-83ae-b38feebe6629","modelid":"84bed8a4-4b9c-41f2-8761-8b0ea94fde60"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("dd819e01-8269-4086-83ae-b38feebe6629")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>


<h2>Saving the Model</h2>
To save the model for use in making real-world predictions, potentially as part of a webapp, we need to freeze the tensorflow graph and transform the variables into constants to maintain the final network. The tutorial [here](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc#.eopkd8pys) walks us through how to accomplish this.


```python
from tensorflow.python.framework import graph_util
def freeze_graph(model):
    # We precise the file fullname of our freezed graph
    output_graph = "/tmp/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "InputData/X,FullyConnected/Softmax"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    
    # We import the meta graph and retrieve a Saver
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = model.net.graph
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # We use a built-in TF helper to export variables to constants
    sess = model.session
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        input_graph_def, # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
```


```python
freeze_graph(model)
```

    INFO:tensorflow:Froze 152 variables.
    Converted 8 variables to const ops.
    607 ops in the final graph.
    


```python
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
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
    return graph
```


```python
grph = load_graph("/tmp/frozen_model.pb")
```


```python
x = grph.get_tensor_by_name('prefix/InputData/X:0')
y = grph.get_tensor_by_name("prefix/FullyConnected/Softmax:0")

# We launch a Session
with tf.Session(graph=grph) as sess:
    # Note: we didn't initialize/restore anything, everything is stored in the graph_def
    y_out = sess.run(y, feed_dict={
        x: [[1]*sequence_chunk_size] 
    })
    print(y_out) # [[ False ]] Yay, it works!
```

    [[ 0.00028193  0.00028239  0.00028253 ...,  0.00028406  0.00028214
       0.00028208]]
    

<h2>Final Recommender</h2>
Using the frozen model, we can predict the most likely subreddits to be of interest to a user by collecting Reddit data for a specific user and provide final recommendations based on the most common subreddits with the highest probabilities from the RNN predictions for each of the subreddit sequence chunks of the user.


```python
from collections import Counter
def collect_user_data(user):
    #Import configuration parameters, user agent for PRAW Reddit object
    config = configparser.ConfigParser()
    config.read('secrets.ini')

    #load user agent string
    reddit_user_agent = config.get('reddit', 'user_agent')
    client_id = config.get('reddit', 'client_id')
    client_secret = config.get('reddit', 'client_api_key')
    #initialize the praw Reddit object
    r = praw.Reddit(user_agent=reddit_user_agent,client_id = client_id,client_secret=client_secret) 
    praw_user = r.get_redditor(user)
    user_data = [(user_comment.subreddit.display_name,
                  user_comment.created_utc) for user_comment in praw_user.get_comments(limit=None)]
    return sorted(user_data,key=lambda x: x[1]) #sort by ascending utc timestamp

def user_recs(user,n_recs=10,chunk_size=sequence_chunk_size):
    user_data = collect_user_data(user)
    user_sub_seq = [vocab.index(data[0]) if data[0] in vocab else 0 for data in user_data]
    non_repeating_subs = []
    for i,sub in enumerate(user_sub_seq):
        if i == 0:
            non_repeating_subs.append(sub)
        elif sub != user_sub_seq[i-1]:
            non_repeating_subs.append(sub)
    user_subs = set([vocab[sub_index] for sub_index in non_repeating_subs])
    sub_chunks = list(chunks(non_repeating_subs,chunk_size))
    user_input = pad_sequences(sub_chunks, maxlen=chunk_size, value=0.,padding='post')
    x = grph.get_tensor_by_name('prefix/InputData/X:0')
    y = grph.get_tensor_by_name("prefix/FullyConnected/Softmax:0")
    with tf.Session(graph=grph) as sess:
        sub_probs = sess.run(y, feed_dict={
            x: user_input
        })
    #select the subreddit with highest prediction prob for each of the input subreddit sequences of the user
    recs = [np.argmax(probs) for probs in sub_probs]
    filtered_recs = [filt_rec for filt_rec in recs if filt_rec not in user_sub_seq]
    top_x_recs,cnt = zip(*Counter(filtered_recs).most_common(n_recs))
    sub_recs = [vocab[sub_index] for sub_index in top_x_recs]
    return sub_recs
```


```python
user_recs("ponderinghydrogen")
```




    ['fantasyfootball', 'PS3']



<h2>The Web App</h2>
Those are all the pieces required to build a functioning subreddit recommender system that users can try! Using Flask, a simple web app can be made taking as input any valid reddit user name and outputting recommendations for that user. A minimal web app doing just that can be interacted with [here](http://ponderinghydrogen.pythonanywhere.com/)

![webapp](documentation/images/wepapp.PNG)

<h2>Final Thoughts</h2>
The model being served in the above webapp is an under-tuned and under-dataed proof-of-concept single layer RNN, but it is still  surprisingly capable of suggesting interesting subreddits to some testers I've had use the app. Nueral Networks really are powerful methods for tackling difficult problems, and with better and better Machine Learning research and tooling being released daily, and increasingly powerful computers, the pool of potential problems solvable by a determined engineer keeps getting larger. I'm looking forward to tackling the next one.
