#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:49:21 2022

@author: hanjiayue
"""


import re
from textblob import TextBlob
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

class nlp:
    
    def __init__(self, filename):
        ''' Constructor '''
        self.M = {}  # stores text specific analysis variables
        self.text = ''
        
    def clean_text(self, t):
        ''' remove everything after the https in the text'''
        split_str = t.split('https', 1)
        substr = split_str[0]
        return substr
            
    def read_text(self, filename):
        ''' Reads given text from file, converts to string and stores '''
        raw_csv = pd.read_csv(filename, names=['brand', 'date', 'code', 'text', 'unlike', 'like'], sep = ',')
        raw_csv['text'] = raw_csv['text'].apply(lambda x: self.clean_text(x))
        for row in raw_csv['text']:
            row = row + '.'
            self.text += row
        return self.text

            
    def sent_tokenize(self):
        ''' Returns list of stripped, tokenized sentences in text '''
        sents = re.split('[.!?]+', self.text.lower())
        return sents
    
    def word_tokenize(self):
        ''' Returns list of stripped, tokenized words in text '''
        words = [re.sub('[^\w\s]', '', word) for word in re.split('\s+', self.text.lower())]
        while("" in words):
            words.remove("")
        return words
    
    
    def sentence_count(self):
        ''' Calculates total sentences in text '''
        sent_count = len(self.sent_tokenize())
        self.M['sentence_count'] = sent_count
        return sent_count

    def total_word_count(self):
        ''' Returns total words in text '''
        total_word_count = len(self.word_tokenize())
        self.M['total_word_count'] = total_word_count
        return total_word_count

    def word_count(self):
        ''' Returns word counts for each text and stores it'''

        word_count = Counter(self.word_tokenize())
        # word_count = [(key, val) for key, val in word_count.items()]
        # word_count.sort(key=lambda x: x[1])

        self.M['word_counts'] = word_count
        return word_count
    
    def avg_sent_len(self):
        ''' Returns and stores average sentence length for given text '''
        avg_slen = round(self.total_word_count() / self.sentence_count(), 3)
        self.M['avg_sentence_length'] = avg_slen
        return avg_slen
    
    def avg_word_len(self):
        ''' Returns and stores average word length for given text excluding punctuation '''
        total_wchar = sum([len(word) for word in self.word_tokenize()])
        avg_wlen = round(total_wchar / self.total_word_count(), 3)
        self.M['avg_word_length'] = avg_wlen
        return avg_wlen
    
    def sentiment(self, per_lines=1, minsub=0.0, maxsub=1.0, minpol=-1.0, maxpol=1.0):
        ''' Gets polarity and subjectivity of text per given lines (int), stores, and returns tuple (pol,sub).
        '''

        # Finds Polarity and subjectivity for a string
        pol, sub = TextBlob(self.text).sentiment
        pol, sub = (round(pol,3), round(sub,3))

        # Sets the Polarity and Subjectivity values to an attribute
        if minpol <= pol <= maxpol and minsub <= sub <= maxsub:
            self.M['sentiment_avg'] = pol,sub
        return pol,sub
    
    def count_syllable(self, word):
        ''' Counts the number of syllables in a word
        https://stackoverflow.com/questions/46759492/syllable-count-in-python '''

        # Sets the list of vowels
        count = 0
        vowels = 'aeiouy'

        # Checks if the first letter is a vowel
        if word[0] in vowels:
            count += 1

        # Iterates through string to see check for consonant followed by vowel
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1

        # Checks if word ends with an e
        if word.endswith('e'):
            count -= 1

        # Sets the minimum count to 1 and then returns it
        return 1 if count < 1 else count
    
    
    def readability(self):
        ''' Assesses Readability using the Gunning Fog Formula '''

        # Finds the variables necessary for Gunning Fog Calculation
        sen_length = self.sentence_count()
        words = self.word_tokenize()
        syllables = [self.count_syllable(word) for word in words]

        # Finds percentage of hard words and then performs gunning fog equation
        hard_word_count = [count for count in syllables if count >= 2]
        hard_word_percent = (len(hard_word_count)/ len(words)) * 100
        return 0.4 * (sen_length + hard_word_percent)

    @staticmethod
    def create_mapping(data, word_lst):
        ''' Create the mapping dictionary for the labels '''

        df = pd.DataFrame(columns=['Source', 'Target', 'Value'])

        for key, value in data.items():
            for word in word_lst:
                df.loc[len(df.index)] = [key, word, value[word]]

        # Creates a sorted list of all the labels
        labels = list(df['Source']) + list(df['Target'])
        labels = sorted(list(set(labels)))

        # Creates a dictionary mapping the labels to assigned values
        codes = list(range(len(labels)))
        code_map = dict(zip(labels, codes))

        # Updates dictionary using the map generated and then returns it
        data = data.replace({'Source': code_map, 'Target': code_map})
        return data, labels


    def wordcount_sankey(self, word_lst=None, K=5):
        ''' Creates a Sankey Diagram to compare the word count of each file '''

        word_count_lst = {}
        word_count_map = {}
        for file in self.contained_files:
            words = getattr(self, file)['word_counts']
            word_count_map[file] = words

            for key, val in words.items():
                word_count_lst[key] = word_count_lst.get(key, 0) + 1


        if not word_lst:
            word_count_lst = [(key, val) for key, val in word_count_lst.items()]
            word_count_lst.sort(key=lambda x: x[1])
            word_lst = [word[0] for word in word_count_lst[-K:]]


        # Updates the data and creates labels
        data, labels = self.create_mapping(word_count_map, word_lst)

        # Creates the link dict for the sankey plot
        link = {'Source': data['Source'], 'Target': data['Target'],
                'Value': data['Value']}

        # Creates the node dict for the sankey plot
        node = {'pad': 100, 'thickness': 10,
                'line': {'color': 'black', 'width': 2},
                'label': labels}

        # Creates the sankey plot and then shows it in browser
        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)
       
    def run_all(self, filename):
        ''' Runs all general analysis methods and returns dictionary with results '''
        gen_methods = [self.read_text(filename), self.avg_sent_len(),self.word_count(),
                       self.avg_word_len(), self.sentiment(per_lines='all'), self.readability()]
        
        for nlp_method in gen_methods:
            nlp_method
            
            return self.M


    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.text

           
    
def main():
   pass
    

if __name__ == '__main__':
    main()

   
