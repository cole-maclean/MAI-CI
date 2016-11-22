import praw
import os
import configparser
import webbrowser
import random
import json
import pandas as pd
import numpy as np
import collections
import string
from nltk.corpus import stopwords
from nltk.stem import *
os.chdir("C:/Users/macle/Desktop/UPC Masters/Semester 2/CI/Final Project")

config = configparser.ConfigParser()
config.read('secrets.ini')

reddit_client_id = config.get('reddit', 'client_id')
reddit_api_key = config.get('reddit', 'api_key')

r = praw.Reddit('reddit recommender by u/upcmaici ver 0..0.1 ')
r.set_oauth_app_info(client_id=reddit_client_id,
                      client_secret=reddit_api_key,
                      redirect_uri='http://cole-maclean.github.io/MAI-CI/')

url = r.get_authorize_url('uniqueKey', 'identity', True)
webbrowser.open(url)

access_information = r.get_access_information('NRQP3Xn0A5sYd-pLj4wxVvnn6Xo')
r.set_access_credentials(**access_information)

authenticated_user = r.get_me()
print(authenticated_user.name, authenticated_user.link_karma)

translate_table = dict((ord(char), None) for char in string.punctuation)
stop = set(stopwords.words('english'))
stemmer = PorterStemmer()
with open('reddit_data.json','r') as data_file:    
    reddit_data = json.load(data_file)
with open('scrapped_users.json','r') as data_file:    
    scrapped_users = json.load(data_file)
for i in range(10):
    print ("scrapping " + str(i) + "th subreddit")
    rand_submission = r.get_random_submission()
    print ("scrapping users from subreddit r/" + rand_submission.subreddit.display_name)
    sub_comments = rand_submission.comments
    if len(sub_comments) >=3:
        rnd_comments = random.sample(sub_comments,min(10,len(sub_comments)))
        for comment in rnd_comments:
            if isinstance(comment, praw.objects.Comment):
                user = comment.author
                if user:
                    if user.name in scrapped_users:
                        print ('user ' + user.name + 'already scraped')
                    else:
                        print ('scrapping data for user ' + user.name)
                        scrapped_users.append(user.name)
                        for user_comment in user.get_comments(limit=None,_use_oauth=False):
                            body = user_comment.body.split()
                            if len(body) >= 10:
                                rand_words = random.sample(body,10)
                            else:
                                rand_words = body
                            clean_comment_words = []
                            for word in rand_words:
                                if len(word) < 45 and word not in stop:
                                    clean_word = word.translate(translate_table)
                                    wrd_stm = stemmer.stem(clean_word)
                                    clean_comment_words.append(wrd_stm)
                            reddit_data.append([user.name,user_comment.subreddit.display_name,user_comment.submission.title,
                                                                              user_comment.created_utc,clean_comment_words])
                    
with open('reddit_data.json','w') as data_file:
    json.dump(reddit_data, data_file)
with open('scrapped_users.json','w') as data_file:
    json.dump(scrapped_users, data_file)