import re
from textblob import TextBlob


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


def count_syllable(word):
    ''' https://stackoverflow.com/questions/46759492/syllable-count-in-python '''

    count = 0
    vowels = 'aeiouy'

    if word[0] in vowels:
        count += 1

    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1

    if word.endswith('e'):
        count -= 1

    return 1 if count == 0 else count


def readability(string):
    ''' Gunning Fog Formula '''

    sen_length = sentence_length(string)

    words = get_words(string)
    syllables = [count_syllable(word) for word in words]

    hard_word_count = [count for count in syllables if count >= 2]
    hard_word_percent = (len(hard_word_count)/ len(words)) * 100

    return 0.4 * (sen_length + hard_word_percent)


def get_polarity(string):
    ''' '''

    blob = TextBlob(string)
    return blob.sentiment.polarity


def get_subjectivity(string):
    ''' '''

    blob = TextBlob(string)
    return blob.sentiment.subjectivity



