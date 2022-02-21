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

class tweet_nlp:
   
    def __init__(self, filename):
        """Constructor"""

        self.data = self.load_text(filename, parser=nlp)
        
    
    @staticmethod
    def _default_parser(filename):
        
        columns = ['brand','date','ID','content','retweet_ct','fav_ct']
        file = pd.read_csv(filename, sep=',',  names=columns)
        text = [ row.lower() +'.' for row in file['content']]
        text = [ TextBlob(row).correct for row in text]
        
        results = {
            'wordcount': Counter(text.split()),
            'sentencecount': len(re.split('[.!?]+', text.lower())),
            'numwords': len(text.split()) }
        
        return results  
    
    def _save_results(self, label, results):
        for k, v in results.items():
            self.data[k][label] = v

    def load_text(self, files, parser=None):

        if type(files) != list:
            files = [files]

        if parser is None:
            results = tweet_nlp._default_parser(files)
        else:
            results = parser(files)

        # A list of common or stop words.  These get filtered from each file automatically
        return results
    
    def polar_sent(self,files,labels):
        
        for i in range(len(labels)):
            for file in files:
                
                sent_word = [len(sent.replace(' ','')) for sent in self.data.sent_tokenize()]
                polar = [TextBlob(sent).sentiment[0] for sent in self.data.sent_tokenize()]
    
                plt.subplot(len(labels), 1, i+1) # (rows, columns, panel number)
                plt.title(labels[i]+' Polarity by Sentence', fontsize=13)
                plt.bar(sent_word, polar, label=labels[i], color = 'c')
                plt.xlabel('Sentence Length')
                plt.xlabel('Polarity Score')

                plt.tight_layout(w_pad=6)
              
    def nlp_seaborn(self, dic, list_files):
        '''graph the average sentence length, polarity, subjectivity, and readability score of all three articles'''
        
        # the values that the graph does not need and need
        del_keys = ['total_word_count', 'sentence_count', 'avg_word_length', 'word_counts', 'text']
        nlp_columns = ['avg_sentence_length','polarity','subjectivity','readability']
        
        # initialize the dataframe
        df = pd.DataFrame(columns = nlp_columns)
        
        # delete the values that we don't need in the dic
        for n in list_files:
            new_dict = dic[n]
            for key in del_keys:
                del new_dict[key]
            
       # generate a tweet sentiment analysis using the dictionaty
        for n in list_files:
            df = df.append(pd.DataFrame(dic[n], columns = nlp_columns, index = [n]))
        
        # graph the figure
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
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
    files = ['car_tweets_AUDI.csv','car_tweets_BMW.csv', 'car_tweets_MERCEDES.csv']
    car_data = tweet_nlp(files)
    car_data.polar_sent(['car_tweets_AUDI.csv', 'car_tweets_BMW.csv'],['Audi','BWM','Benz'])
    car_data.nlp_seaborn(tweet_dict, files)
