"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()


def get_subject_verb_pairs(formatted_corpus,subjname):
    all_pairs = dict()
    for sentence in formatted_corpus:
        for word in sentence:
            #Call the subject function (in swedish) for pairs : SS
            if word['deprel'] == subjname:
                #Create a pair of
                # 1. The subject
                # 2. The verb, which is linked as the head.
                pair = (word['form'].lower(), sentence[int(word['head'])]['form'].lower())

                #Increase count of pair / Create the key and set cound to 1
                if pair in all_pairs.keys():
                    all_pairs[pair] += 1
                else:
                    all_pairs[pair] = 1
    return all_pairs


def get_total_nbr_subject_verb_pairs(pairs):
    return sum(pairs.values())

def get_subject_verb_triples(formatted_corpus, objname, subjname):
    all_triples = dict()
    for sentence in formatted_corpus:
        for word in sentence:
            #Call the subject function (in swedish) for triples: OO
            if word['deprel'] == objname:
                #Create a pair of

                obj_word = word['form'].lower()

                for subj_word in sentence:

                    #They need to point at the same head! AND ofcourse have the right rule.
                    if subj_word['head'] == word['head'] and subj_word['deprel'] == subjname:

                        triple = (subj_word['form'].lower(), sentence[int(subj_word['head'])]['form'].lower(), obj_word)

                        #Increase count of pair / Create the key and set count to 1
                        if triple in all_triples.keys():
                            all_triples[triple] += 1
                        else:
                            all_triples[triple] = 1
    return all_triples

def get_all_pairs(formatted_corpus,subjname):

    pairs = get_subject_verb_pairs(formatted_corpus,subjname)
    print("Total number of pairs: " + str(get_total_nbr_subject_verb_pairs(pairs)))

    # Sort the valus
    pairs_sorted_values = sorted(pairs.values(), reverse=True)

    # Sort the keys
    pairs_sorted_keys = sorted(pairs, key=pairs.get, reverse=True)

    for i in range(5):
        try:
            print(
                '(' + pairs_sorted_keys[i][0] + ',' + pairs_sorted_keys[i][1] + ')' + ', # ' + str(pairs_sorted_values[i]))
        except:
            print("no more pairs...")
            break

def get_all_triples(formatted_corpus, objname, subjname):
    triples = get_subject_verb_triples(formatted_corpus,objname, subjname)
    print("Total number of pairs: " + str(get_total_nbr_subject_verb_pairs(triples)))

    # Sort the valus
    pairs_sorted_values = sorted(triples.values(), reverse=True)

    # Sort the keys
    pairs_sorted_keys = sorted(triples, key=triples.get, reverse=True)

    for i in range(5):
        try:
            print(
                '(' + pairs_sorted_keys[i][0] + ',' + pairs_sorted_keys[i][1] + ','+pairs_sorted_keys[i][2] + ')' + ', # ' + str(pairs_sorted_values[i]))
        except:
            print("no more pairs...")
            break
def get_all_languages():

    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    files = get_files('../../../corpus/treebank/ud-treebanks-v2.2/', 'train.conllu')
    for train_file in files:
        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_u)
        print(train_file, len(formatted_corpus))
        print(formatted_corpus[0])
        get_all_pairs(formatted_corpus,"nsubj")
        #get_all_triples(formatted_corpus,"obj","nsubj")


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    train_file = '../swedish_talbanken05_train.conll.txt'
    # train_file = 'test_x'
    test_file = '../swedish_talbanken05_test.conll.txt'

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)
    #print(train_file, len(formatted_corpus))
    #print(formatted_corpus[0])

    #get_all_pairs(formatted_corpus, "SS")

    #get_all_triples(formatted_corpus, "OO", "SS")

    get_all_languages()

