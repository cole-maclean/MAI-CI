import praw
import configparser
import random
import json
import pandas as pd
import numpy as np
import collections
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys

#Import configuration parameters, user agent for PRAW Reddit object
config = configparser.ConfigParser()
config.read('secrets.ini')

#load user agent string
reddit_user_agent = config.get('reddit', 'user_agent')
client_id = config.get('reddit', 'client_id')
client_secret = config.get('reddit', 'client_api_key')

#main data scrapping script
def scrape_data(n_scrape_loops = 10,dataset='train',nlp_data=False):

    """This is the main function that runs the scrapping functionality through praw. The random_submission method is called
    n_submissions times to obtain n random submissions. After receiving a random submission, the .comments method on the submission
    object is used to get the comments in the submission. The minimum number of comments for all the comments in the submission or max_sub_comments
    are randomly selected, and the author of each comment is obtained. The comment history of each randomly selected comment author is parsed, and 
    comment, submission and subreddit data are collected and stored.

    parameters:

    n_submissions - the number of random submissions to select in finding random comments and comment authors to parse their comment history from

    max_sub_comments - the maximum number of comments to randomly select from a random submission for author search and comment history parsing

    dataset - determines if the data is stored into the training or testing user dataset

    nlp_data - set to true to include scrapping data for NLP analysis (user comment words and commented submission titles)

    """

    if nlp_data == True:
        nlp_flag = 'nlp'
    else:
        nlp_flag = ''
        r = praw.Reddit(user_agent=reddit_user_agent,client_id = client_id,client_secret=client_secret) #initialize the praw Reddit object
        translate_table = dict((ord(char), None) for char in string.punctuation) #dictionary lookup for removing punctuation characters in comment and submission title data
        stop = set(stopwords.words('english')) #english stop words to filter out non-useful words before storing
        stemmer = PorterStemmer() #stemming for similiar word aggregation
        with open('data/' + dataset + '_reddit_data' + nlp_flag + '.json','r') as data_file:    
            reddit_data = json.load(data_file)
        with open('data/scrapped_users.json','r') as data_file:    
            scrapped_users = json.load(data_file)
        for scrape_loop in range(n_scrape_loops):
            try:
                all_comments = r.get_comments('all')
                print ("Scrape Loop " + str(scrape_loop))
                for cmt in all_comments:
                    user = cmt.author        
                    if user:
                        print ("Collecting Data for User " + user.name)
                        if user.name in scrapped_users: #check if the users data has already been parsed and skip parsing if True
                            print ('user ' + user.name + ' already scraped')
                        else:
                            scrapped_users.append(user.name) #update already scrapped user cache with currently scraped user
                            for user_comment in user.get_comments(limit=None):
                                if nlp_data == True:
                                    body = user_comment.body.split()
                                    #filter out all but the min of 10 words or the total body word count from the comments body to reduce the dataset size.
                                    rand_words = random.sample(body,min(10,len(body)))
                                    clean_comment_words = []
                                    for word in rand_words:
                                        #perform stop word, punctuation and stemming cleaning on comment body words
                                        if len(word) < 45 and word not in stop:
                                            clean_word = word.translate(translate_table)
                                            clean_comment_words.append(clean_word)
                                    #append username, subreddit name, submission title, comment utc timestamp and cleaned random comment body words to dataset
                                    reddit_data.append([user.name,user_comment.subreddit.display_name,user_comment.link_title.split(' '),
                                                                                      user_comment.created_utc,clean_comment_words])
                                else:
                                    reddit_data.append([user.name,user_comment.subreddit.display_name,
                                                  user_comment.created_utc])
            except Exception as e:
                print(e)
    #dump scrapped dataset to script                   
    with open('data/' + dataset + '_reddit_data' + nlp_flag + '.json','w') as data_file:
        json.dump(reddit_data, data_file)
    with open('data/scrapped_users.json','w') as data_file:
        json.dump(scrapped_users, data_file)

if __name__ == "__main__":
    scrape_data(int(sys.argv[1]), sys.argv[2],sys.argv[3])