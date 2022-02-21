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
       

def main():

    car_data = tweet_nlp(['car_tweets_AUDI.csv',
                     'car_tweets_BMW.csv','car_tweets_MERCEDES.csv'])
    car_data.polar_sent(['car_tweets_AUDI.csv', 'car_tweets_BMW.csv'],['Audi','BWM','Benz'])
