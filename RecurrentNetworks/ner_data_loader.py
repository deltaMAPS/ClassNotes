import sys
import os
import time
from keras.utils.np_utils import to_categorical


##Symbol for unknown word
UNK = "UNK"


##https://programminghistorian.org/en/lessons/counting-frequencies#python-dictionaries
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()

    aux = [entry for _,entry in aux]; ##keep only the words

    return aux

"""
Function that loads the raw .txt data for named entity recognition.

Data format:

The first line of the file is a "-DOCSTART- -X- -X- O", we will skip this line.
Sentences are separated by an empty line ( \n character).
Each entry in a sentence is separated by a space and the format is [word pos_tag pos_tag2 named_entity]

For this exercise we will be focusing on named entity recognition. The available tags are:
1. B-ORG (Organization)
2. I-ORG
3. B-LOC (Location)
4. I-LOC
5. B-PER (Person)
6. I-PER
7. B-MISC (Miscellaneous)
8. I_MISC
9. O (Other)
B/I determine the order of the tags 
Example
John  lives in New    York
B-PER O     O  B-LOC  I-LOC

Inputs:
    filename (string): full path to the data file
    min_length (int): sentences with length < min_length are discarded
    max_length (int): sentences with length > max_length are discarder
Returns:
    sentences (list): List of lists(sentences), each sub-list holds the words of a sentence
    sentence_labels (list): List of lists(sentence tags), each sub-list holds the ner tags of the sentence
"""
def load_data(filename, min_length = -1, max_length = sys.maxint, keep_top = 20000):

    print('Parsing data');

    data = open(filename).readlines();

    ##store sequences here as list of lists
    sentences = [];
    ##store sequence labels here as list of lists
    sentence_labels = [];
    ##unique words and their frequency here
    word_freqs = {};
    ##unique labels and their frequency here
    label_freqs = {}

    ##min & max sentence length
    min_sentence_length = sys.maxint;
    max_sentence_length = -1;


    ##start reading the lines, skipping the first one which is the header and the second one which is an empty line
    current_sentence = []; 
    current_sentence_labels = [];


    for i in range(2,len(data)):
        if(data[i] == '\n'): ##new line, store old stuff and move on
            if(len(current_sentence) >= min_length and len(current_sentence) <= max_length):
                if(len(current_sentence) > max_sentence_length):
                    max_sentence_length = len(current_sentence);
                if(len(current_sentence) < min_sentence_length):
                    min_sentence_length = len(current_sentence);
                sentences.append(current_sentence);
                current_sentence = [];
                sentence_labels.append(current_sentence_labels);
                current_sentence_labels = [];
            else: ##ignore
                current_sentence = [];
                current_sentence_labels = [];
        else:
            ##split 
            args = data[i].split(' '); ##args[0] -> word, args[1] -> postag
            ##frequency updates
            if(args[0] in word_freqs):
                word_freqs[args[0]] +=1.0;
            else:
                word_freqs[args[0]] = 1.0;
            if(args[3] in label_freqs):
                label_freqs[args[3]] += 1.0;
            else:
                label_freqs[args[3]] = 1.0; 
            current_sentence.append(args[0]);
            current_sentence_labels.append(args[3]);


    print('Done.');
    print('Sentences: '+str(len(sentences)));
    print('Max sentence length: '+str(max_sentence_length));
    print('Min sentence length: '+str(min_sentence_length));
    print('Unique words on corpus: '+str(len(word_freqs)) );
    print('Unique labels on corpus: '+str(len(label_freqs)));

        
    if(keep_top != None):
        print('Pruning non-frequent words, keeping top '+str(keep_top));
        ##sort word freq dictionary and keep the top k frequent
        most_freq_words = sortFreqDict(word_freqs)[0:keep_top];
    
        ##replace
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if(not (sentences[i][j] in most_freq_words)):
                    sentences[i][j] = UNK; ##replace with the unknooooowwwnnnn 
   
    return sentences, sentence_labels;

"""
Function that constructs word and label dictionaries for sentences and sentence labels.
Dictionaries map a word/label to an integer id (and vice versa). 

In this example word ids start from 2. We keep 0 for the pad symbol and 1 for the UNK symbol
Label ids start from 1 keeping 0 for the pad symbol.

Inputs:
    sentences (list): List of lists(sentences), each sub-list holds the words of a sentence
    sentence_labels (list): List of lists(sentence tags), each sub-list holds the ner tags of the sentence
Returns:
    word_to_ind(dict): dictionary mapping a word to an integer index
    ind_to_word(dict): dictionary mapping an integer index to a word
    label_to_ind(dict): dictionary mapping a label to an integer index
    ind_to_label(dict): dictionary mapping an integer index to a label
"""
def get_dictionaries(sentences, sentence_labels):
    ##construct a dictionary mapping words & labels to integers
    start_ind_words = 2;
    start_ind_labels = 1;

    word_to_ind = {};
    label_to_ind = {};

    ##first things first, add the unknown symbol
    word_to_ind[UNK] = 1;

    for sentence in sentences:
        for word in sentence:
            if(word not in word_to_ind):
                word_to_ind[word] = start_ind_words;
                start_ind_words+=1;
    for sentence in sentence_labels:
        for label in sentence:
            if(label not in label_to_ind):            
                label_to_ind[label] = start_ind_labels;
                start_ind_labels+=1;
    ind_to_word = {ind:word for word,ind in word_to_ind.iteritems()};
    ind_to_label = {ind:label for label,ind in label_to_ind.iteritems()};
    
    print('Unique words in data: '+str(len(word_to_ind)));
    print('Unique labels in data: '+str(len(label_to_ind)));
    
    return word_to_ind, ind_to_word, label_to_ind, ind_to_label;

"""
Function that transforms a list of lists holding labels/words to a list of
lists holding the indexed (integer) representation of words. 

Inputs:
    data(list): List of lists (sentences or sentence labels), usually the output of load_data(...)
    data_to_ind(dict): dictionary mapping words or labels to their corresponding integer index, usually the output of get_dictionaries(...)
Returns:
    indexed_data: List of lists containing the corresponding integer representation of the strings in data
"""
def to_index(data, data_to_ind):
    indexed_data = [ [data_to_ind[x] for x in sentence] for sentence in data];
    return indexed_data;
    
    


if __name__ == "__main__":
	fname = "train.txt"
	sentences, sentence_labels = load_data(fname,min_length=5,max_length=64)

