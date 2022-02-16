import re


def get_words(string):
    ''' '''

    words = re.split('[.!? ]+', string)
    return [val for val in words if val != '']


def word_count(string):
    ''' '''

    return len(get_words(string))


def word_length(string):
    ''' '''

    words = get_words(string)
    word_lengths = [len(word) for word in words]

    return round(sum(word_lengths)/len(word_lengths))


def get_sentences(string):

    sentences = re.split('[.!?]+', string)
    return [val for val in sentences if val != '']


def sentence_count(string):
    ''' '''

    return len((get_sentences(string)))


def sentence_length(string):
    ''' '''

    sentences = get_sentences(string)
    sentence_lengths = [len(stc.split()) for stc in sentences]

    return round(sum(sentence_lengths)/len(sentence_lengths))








