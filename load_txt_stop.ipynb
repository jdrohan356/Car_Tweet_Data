class texttt:
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
    #for name in filename:
        columns = ['brand','date','ID','content','retweet_ct','fav_ct']
        file = pd.read_csv(filename, sep=',',  names=columns)
    return file
    import requests
    def load_stop_words(stopfile=None):

        #stopf = input('Would you import a stopwords file? Sumbit the file name here pls:')

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
