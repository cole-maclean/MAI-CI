# A Recurrent Neural Network Based Subreddit Recommendation System

This repository contains the code and data exploration for the creation of a Subreddit recommender system based on tensorflow and the tflearn wrapper using a
Recurrent Neural Network architecture.

Training data can be generated using the reddit_scrapper.py script. Note - a configuration file with reddit API keys will be required. The file needs to be placed in the 
SUbRecommender folder and consits of:

[reddit]  
api_key: key  
client_id: id  
client_api_key: client key  
redirect_url: redir url  
user_agent: subreddit-recommender by /u/upcmaici v 0.0.1   

Data is saved under the SubRecommender folder with the following tree structure (some folders may need to be created):

SubRecommender/data
SubRecommender/data/training_sequences


An exploration of the data and development of the model can be found in the jupyter notebook SubRecommender/EDA Notebook.ipynb

The final model is stored under SubRecommender/models

The final embedding can be interacted with in a browser and is stored as SubRecommender\embedding.html

### Install requirements

```
pip install -r requirements.txt
```

### to run

```
jupyter notebook
```

Go to `SubRecommender/` folder and open `EDA Notebook.ipynb` jupyter notebook and go through it.

