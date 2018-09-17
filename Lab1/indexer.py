#!/usr/bin/env python3

import regex as re
import sys
import os
import pickle
from math import log10
import numpy as np

# See p. 168-173 in book draft 2018-09-03

tf_idf = []


def index_words(text):
    """
    Takes a text and returns a list of all words and their corresponding placement in the text
    :param text:
    :return: a dict where key = word, value = position in text
    """
    text = text.lower()
    words = re.finditer(r'\p{L}+', text)

    index = {}

    for word in words:
        w = word.group()  # make a string of the regex object so we can use it
        if w in index:
            index[w].append(word.start())  # .start() gets starting position of match
        else:
            index[w] = [word.start()]

    return index


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def create_index(directory):
    """
    Creates and returns the master index from all files in a given directory
    """
    master_index = {}

    files = get_files(directory, 'txt')
    print("Inlästa filer: ", files, "\n\n")

    for f in files:
        f_o = open(directory + "/" + f, "r")
        corpus_text = f_o.read()
        word_index = index_words(corpus_text)

        # create new index file for each file

        save_obj(word_index, 'Index/' + f[:-4])

        for word in word_index:
            if word in master_index:
                master_index[word][f] = word_index[word]
            else:
                master_index[word] = {}
                master_index[word][f] = word_index[word]
        # print(master_index)

        # Save the master index file
        save_obj(master_index, "master_index")


def save_obj(obj, name):
    """
    Saves an object as a pickle file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    """
    Loads and return an object from a pickle file
    """
    if name[-3:] == "pkl":
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


def tfidf(freq_of_word, freq_all_words, tot_nbr_docs, docs_with_word):
    """
    Calculates and returns TF x IDF
    """
    tf = freq_of_word / freq_all_words
    idf = log10(tot_nbr_docs / docs_with_word)
    return tf * idf


def calculate_vectors():
    master_index = load_obj("master_index")

    files = get_files("Index", "pkl")

    # create a dict with key = filename, value = {word : occurences in current file}

    file_info = {
        f.replace("pkl", "txt"): load_obj("Index/" + f) for f in files
    }

    # create dict with key = filename, value = {}
    vectors = {f.replace("pkl", "txt"): {} for f in files}
    vector_files = {}

    tot_nbr_docs = len(files)
    tot_word_freq = word_frequency(file_info)

    for word in master_index:
        # get nbr of docs containing word by searching in master index for all docs containing word
        docs_with_word = len(master_index[word])

        for f in file_info:
            # count all words in file f
            freq_all_words = tot_word_freq[f[:-3] + "txt"]
            try:
                # if word exists, calculate tdidf
                word_freq = len(file_info[f][word])
                vectors[f][word] = tfidf(word_freq, freq_all_words, tot_nbr_docs, docs_with_word)
            except KeyError:
                # if not, then return 0
                vectors[f][word] = tfidf(0, freq_all_words, tot_nbr_docs, docs_with_word)
            vector_files[f] = vectors[f]

    return vectors


def word_frequency(files):
    total = {}
    for f in files:
        total[f] = 0
        for i in files[f]:
            w_freq = len(files[f][i])
            total[f] = total[f] + w_freq
    return total


def cosine_similarity(v1, v2):
    """
    Takes 2 vectors v1, v2 and returns the cosine similarity according
	to the definition of the dot product
	"""

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def get_matrix(v):
    """
    Compares all documents an returns the results in a matrix
    """
    vectors = v
    length = len(vectors)
    matrix = np.identity(length)
    files = list(vectors.keys())
    #get Documents
    for i in range(length):
        print(i + 1, ". ", files[i])  # print index for each file
        for j in range(length):
            # jämför dokument i med dokument j och räkna ut deras cosine siilarity
            # -> fyll upp matrisen plats (i,j) med det värdet
            # för att få dokument di och dj på rätt format
            di = list(vectors[files[i]].values())
            dj = list(vectors[files[j]].values())
            matrix[i][j] = cosine_similarity(di, dj)
    #prettyprint
    np.set_printoptions(precision=2, suppress=True)
    print("\n\n", matrix)


if __name__ == '__main__':
    create_index("Selma")
    # print(tfidf(1,2,3,4))
    v = calculate_vectors()

    # Test
    #print(v['bannlyst.txt']['känna'])
    #print(v['bannlyst.txt']['gås'])
    #print(v['bannlyst.txt']['nils'])
    #print(v['bannlyst.txt']['et'])

    #print(v['herrgard.txt']['känna'])
    #print(v['herrgard.txt']['gås'])
    #print(v['herrgard.txt']['nils'])
    #print(v['herrgard.txt']['et'])

    #print(v['jerusalem.txt']['känna'])
    #print(v['jerusalem.txt']['gås'])
    #print(v['jerusalem.txt']['nils'])
    #print(v['jerusalem.txt']['et'])

    # print(v['nils.txt']['känna'])
    # print(v['nils.txt']['gås'])
    # print(v['nils.txt']['nils'])
    # print(v['nils.txt']['et'])
    get_matrix(v)
