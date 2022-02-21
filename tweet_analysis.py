import requests
import numpy as np

class tweet_nlp:
    from collections import Counter, defaultdict

    def __init__(self):
        """Constructor"""
        self.data = defaultdict(dict)
    
    @staticmethod
    def _default_parser(filename):
        results = {
            'wordcount': Counter(text_con.split()),
            'numwords': len(text_con.split())
        }
        return results


    def load_text(self, filename_ls, label=None, parser=None):
        if parser is None:
            for file in filename_ls:
                results = texttt._default_parser(file)
        else:
            for file in filename_ls:
                results = parser(file)

        if label is None:
            label = filename

        self._save_results(label, results)
        
        
    def read_text(filename, parser=None):
    
        columns = ['brand','date','ID','content','retweet_ct','fav_ct']
        file = pd.read_csv(filename, sep=',',  names=columns)
    return file
    
    def load_stop_words(stopfile=None):
    
        if stopfile is None:
            stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
            stopwords = stopwords_list.decode().splitlines()
            text = ' '.join([word for word in text_con.split() if word not in stopwords])
        else:
            with open(stopfile) as f:
                for line in f:
                    stopf = [line.split(',') for line in f]
            text = ' '.join([word for word in text_con.split() if word not in stopf])
        # A list of common or stop words.  These get filtered from each file automatically 
        return text

    def polarity_by_sent(files):
        
        # create the first of two panels and set current axis
        plt.subplot(3, 1, 1) # (rows, columns, panel number)
        plt.title('Audi Polarity by Sentence', fontsize=13)
        plt.bar(sent_word_audi, polar_audi, label='Audi')

        # create the second panel and set current axis
        plt.subplot(3, 1, 2)
        plt.title('BMW Polarity by Sentence', fontsize=13)
        plt.bar(sent_word_bmw, polar_bmw, label='BMW');

        plt.subplot(3, 1, 3)
        plt.title('Benz Polarity by Sentence', fontsize=13)
        plt.bar(sent_word_benz, polar_benz, label='Benz');
        plt.figlegend(loc='upper right', ncol=1, labelspacing=0.3, fontsize=8, bbox_to_anchor=(1.11, 0.9))
        plt.tight_layout(w_pad=6)
