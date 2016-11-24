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

config = configparser.ConfigParser()
config.read('secrets.ini')

reddit_user_agent = config.get('reddit', 'user_agent')

def scrape_data(n_subreddits = 10, max_sub_comments = 10):
    r = praw.Reddit(user_agent=reddit_user_agent)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    stop = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    with open('reddit_data.json','r') as data_file:    
        reddit_data = json.load(data_file)
    with open('scrapped_users.json','r') as data_file:    
        scrapped_users = json.load(data_file)
    for i in range(n_subreddits):
        print ("scrapping " + str(i) + "th subreddit")
        rand_submission = r.get_random_submission()
        print ("scrapping users from subreddit r/" + rand_submission.subreddit.display_name)
        sub_comments = rand_submission.comments
        if len(sub_comments) >=3:
            rnd_comments = random.sample(sub_comments,min(max_sub_comments,len(sub_comments)))
            for comment in rnd_comments:
                if isinstance(comment, praw.objects.Comment):
                    user = comment.author
                    if user:
                        if user.name in scrapped_users:
                            print ('user ' + user.name + 'already scraped')
                        else:
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
                                reddit_data.append([user.name,user_comment.subreddit.display_name,user_comment.link_title.split(' '),
                                                                                  user_comment.created_utc,clean_comment_words])
                        
    with open('reddit_data.json','w') as data_file:
        json.dump(reddit_data, data_file)
    with open('scrapped_users.json','w') as data_file:
        json.dump(scrapped_users, data_file)

def rollup_data():
    with open('reddit_data.json','r') as data_file:    
        reddit_data = json.load(data_file)
    df = pd.DataFrame(reddit_data,columns=['user','subreddit','submission','utc_stamp','rnd_words'])
    f = {'subreddit':['count'], 'utc_stamp':[np.min,np.max],'submission': 'sum','rnd_words': 'sum'}
    grouped = df.groupby(['user','subreddit']).agg(f)
    return grouped

if __name__ == "__main__":
    scrape_data(int(sys.argv[1]), int(sys.argv[2]))