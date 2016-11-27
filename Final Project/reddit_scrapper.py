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

#main data scrapping script
def scrape_data(n_submissions = 10, max_sub_comments = 10):

    """This is the main function that runs the scrapping functionality through praw. The random_submission method is called
    n_submissions times to obtain n random submissions. After receiving a random submission, the .comments method on the submission
    object is used to get the comments in the submission. The minimum number of comments for all the comments in the submission or max_sub_comments
    are randomly selected, and the author of each comment is obtained. The comment history of each randomly selected comment author is parsed, and 
    comment, submission and subreddit data are collected and stored.

    parameters:

    n_submissions - the number of random submissions to select in finding random comments and comment authors to parse their comment history from

    max_sub_comments - the maximum number of comments to randomly select from a random submission for author search and comment history parsing

    """
    r = praw.Reddit(user_agent=reddit_user_agent) #initialize the praw Reddit object
    translate_table = dict((ord(char), None) for char in string.punctuation) #dictionary lookup for removing punctuation characters in comment and submission title data
    stop = set(stopwords.words('english')) #english stop words to filter out non-useful words before storing
    stemmer = PorterStemmer() #stemming for similiar word aggregation
    with open('reddit_data.json','r') as data_file:    
        reddit_data = json.load(data_file)
    with open('scrapped_users.json','r') as data_file:    
        scrapped_users = json.load(data_file)
    for i in range(n_submissions):
        print ("scrapping " + str(i) + "th subreddit")
        rand_submission = r.get_random_submission()
        print ("scrapping users from subreddit r/" + rand_submission.subreddit.display_name)
        sub_comments = rand_submission.comments
        if len(sub_comments) >=3: #ensure submission has > 3 comments to filter out submissions with 0 comments, and submissions only containing AutoModerator posts
            rnd_comments = random.sample(sub_comments,min(len(sub_comments),max_sub_comments)) #select randon number of the min of total number of comments in submission or max_sub_comments
            for comment in rnd_comments:
                if isinstance(comment, praw.objects.Comment): #check that return object is a comment object (sometimes a a MoreComment object is returned, which are not currently parsed)
                    user = comment.author
                    if user:
                        if user.name in scrapped_users: #check if the users data has already been parsed and skip parsing if True
                            print ('user ' + user.name + ' already scraped')
                        else:
                            scrapped_users.append(user.name) #update already scrapped user cache with currently scraped user
                            for user_comment in user.get_comments(limit=None,_use_oauth=False):
                                body = user_comment.body.split()
                                #filter out all but the min of 10 words or the total body word count from the comments body to reduce the dataset size.
                                rand_words = random.sample(body,min(10,len(body)))
                                clean_comment_words = []
                                for word in rand_words:
                                    #perform stop word, punctuation and stemming cleaning on comment body words
                                    if len(word) < 45 and word not in stop:
                                        clean_word = word.translate(translate_table)
                                        wrd_stm = stemmer.stem(clean_word)
                                        clean_comment_words.append(wrd_stm)
                                #append username, subreddit name, submission title, comment utc timestamp and cleaned random comment body words to dataset
                                reddit_data.append([user.name,user_comment.subreddit.display_name,user_comment.link_title.split(' '),
                                                                                  user_comment.created_utc,clean_comment_words])
    #dump scrapped dataset to script                   
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