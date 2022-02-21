#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:14:44 2022

@author: tianyang
"""

import requests
import numpy as np
import re
import pandas as pd
from textblob import TextBlob
from textblob import Word
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from NLP import nlp

class tweet_nlp:
   
    def __init__(self):
        """Constructor"""
        self.data = defaultdict(dict)
    
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
             
    def load_text_stop(self, file, label=None, parser=None, stopfile=None):
    
        if stopfile is None:
            stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
            stopwords = stopwords_list.decode().splitlines()
            text = ' '.join([word for word in nlp.text if word not in stopwords])
            
        else:
            with open(stopfile) as f:
                for line in f:
                    stopf = [line.split(',') for line in f]
            text = ' '.join([word for word in nlp.text.split() if word not in stopf])
            
        if parser is None:
                results = tweet_nlp._default_parser(file)
        else:
                results = parser(file)
        if label is None:
                label = file
        self._save_results(label, results)
        
        # A list of common or stop words.  These get filtered from each file automatically 
        #return text
'''def load_text(self, file, label=None, parser=None):
        
        stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
        stopwords = stopwords_list.decode().splitlines()
        text = ' '.join([word for word in nlp.text if word not in stopwords])
        
        if parser is None:
            results = tweet_nlp._default_parser(file)
        else:
            results = parser(file)
        if label is None:
            label = file
        self._save_results(label, results)'''
