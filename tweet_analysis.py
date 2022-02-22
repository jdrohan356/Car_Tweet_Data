#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:14:44 2022

@author: tianyang
"""

from NLP import nlp

from textblob import TextBlob
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np
import re
from textblob import Word
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class tweet_nlp:
   
    def __init__(self, filename,stopfile = None):
        """Initiate variable of stopfile and data for tweet_nlp"""

        self.stopfile = stopfile
        self.data = self.load_text(filename, parser=nlp)

    
    @staticmethod
    def _default_parser(filename):
        '''read in header column of twitter posts''' 
        columns = ['brand','date','ID','content','retweet_ct','fav_ct']
        file = pd.read_csv(filename, sep=',',  names=columns)
        # Lowercase all sentences and add periods to the end to be considered
        # as a complete sentence
        text = [ row.lower() +'.' for row in file['content']]
        # Correct grammar and spelling mistakes in sentences for three files
        text = [ TextBlob(row).correct for row in text]
        
        results = {
            'wordcount': Counter(text.split()),
            'sentencecount': len(re.split('[.!?]+', text.lower())),
            'numwords': len(text.split()) }
        return results  
            
    def load_text(self, files, parser=None):
        '''load texts with parser and stopfile if user provides them, 
        otherwise work with default parser and stopfile'''
        if type(files) != list:
            files = [files]
        if parser is None:
            results = tweet_nlp._default_parser(files)
        else:
            results = parser(files,self.stopfile)
        return results
    
    @staticmethod
    def create_mapping(data, word_lst):
        ''' Create the mapping dictionary for the labels '''

        df = pd.DataFrame(columns=['source', 'target', 'value'])

        for key, value in data.items():
            for word in word_lst:
                df.loc[len(df.index)] = [key, word, value[word]]

        # Creates a sorted list of all the labels
        labels = list(df['source']) + list(df['target'])
        labels = sorted(list(set(labels)))

        # Creates a dictionary mapping the labels to assigned values
        codes = list(range(len(labels)))
        code_map = dict(zip(labels, codes))

        # Updates dictionary using the map generated and then returns it
        data = df.replace({'source': code_map, 'target': code_map})
        return data, labels

    def wordcount_sankey(self, word_lst=None, K=5):
        ''' Creates a Sankey Diagram to compare the word count of each file'''

        word_count_lst = {}
        word_count_map = {}
        # Maps word counts to appropriate file
        for key, value in self.data.contained_files.items():
            words = value['word_counts']
            word_count_map[key] = words
            
            for key, val in words.items():
                word_count_lst[key] = word_count_lst.get(key, val) + val
        # Creates a word list if user does not customize words to be shown
        if not word_lst:
            word_count_lst = [(key, val) for key, val 
                              in word_count_lst.items()]
            word_count_lst.sort(key=lambda x: x[1])
            word_lst = [word[0] for word in word_count_lst[-K:]]
        
        # Updates the data and creates labels
        data, labels = tweet_nlp.create_mapping(word_count_map, word_lst)
        
        # Creates the link dict for the sankey plot
        link = {'source': data['source'], 'target': data['target'],
                'value': data['value']}

        # Creates the node dict for the sankey plot
        node = {'pad': 100, 'thickness': 10,
                'line': {'color': 'black', 'width': 2},
                'label': labels}

        # Creates the sankey plot and then shows it in browser
        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)
        #fig.draw()
        
        
    def polar_sent(self,labels):
        '''Plot distribution of polarity scores by sentence length 
        for three files'''
        
        for i in range(len(labels)):
            # Create list containing all setence lengths in one file    
            sent_word = [len(sent.replace(' ','')) for 
                         sent in self.data.contained_files[labels[i]]['Sents']]
            # Create polarity score of each sentence in one file
            polar = [TextBlob(sent).sentiment[0] for 
                     sent in self.data.contained_files[labels[i]]['Sents']]
            # rows, columns, panel number
            plt.subplot(len(labels), 1, i+1) 
            plt.title(labels[i]+' Polarity by Sentence', fontsize=13)
            plt.bar(sent_word, polar, label=labels[i], color = 'c')
            plt.tight_layout(w_pad=6)
        
        plt.xlabel('Sentence Length')
        plt.ylabel('Polarity Score')
        
    def nlp_seaborn(self, dic, list_files):
        '''graph the average sentence length, polarity, subjectivity, 
        and readability score of all three articles'''
        
        # The values that the graph does not need and need
        del_keys = ['total_word_count', 'sentence_count', 
                    'avg_word_length', 'word_counts', 'text']
        nlp_columns = ['avg_sentence_length','polarity',
                       'subjectivity','readability']
        
        # Initialize the dataframe
        df = pd.DataFrame(columns = nlp_columns)
        
        # Delete the values that we don't need in the dic
        for n in list_files:
            new_dict = dic[n]
            for key in del_keys:
                del new_dict[key]
            
       # Generate a tweet sentiment analysis using the dictionaty
        for n in list_files:
            df = df.append(pd.DataFrame(dic[n], 
                                        columns = nlp_columns, index = [n]))
        
        # Graph the figure
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,
                                                figsize=(10, 10), sharex=True)
        x = ['Audi','BMW','Mercedes']
        y1 = df.avg_sentence_length
        sns.barplot(x=x,y=y1,ax=ax1, palette='RdPu')
        ax1.set_ylabel('Average Sentence Length')
        ax1.set_title('Comparison of Text Structural Measures')
        ax1.set_xlabel('')
        
        y2 = df.readability
        sns.barplot(x=x,y=y2,ax=ax2, palette='BuPu')
        ax2.set_ylabel('Readability')
        ax2.set_xlabel('')
        
        y3 = df.subjectivity
        sns.barplot(x=x,y=y3,ax=ax3, palette='BuPu') 
        ax3.set_ylabel('Subjectivity') 
        ax3.set_xlabel('')
        
        y4 = df.polarity 
        sns.barplot(x=x,y=y4,ax=ax4, palette='BuPu')
        ax4.set_ylabel('Polarity')
        ax4.set_xlabel('')
        
        sns.despine(bottom=False)
        plt.tight_layout(h_pad=1)

        
       

def main():
    car_data = tweet_nlp(['car_tweets_AUDI.csv',
                     'car_tweets_BMW.csv','car_tweets_MERCEDES.csv'])
    
    labels = ['AUDI','BMW','MERCEDES']
    tweet_dict = car_data.data.contained_files
    
    # Sankey Diagram of wordcount and random words
    car_data.wordcount_sankey(['hi','good','share','talk','run'],K=5)
    # Subplots presenting polarity score by sentence lengths
    car_data.polar_sent(['AUDI','BMW','MERCEDES'])
    # Seaborn comparison of average sentence length, polarity, subjectivity, 
    # and readability assessment
    car_data.nlp_seaborn(tweet_dict, labels)
   


if __name__ == '__main__':
    main()
