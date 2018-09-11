import regex as re
import sys
import os
import pickle

def tokenizeUnique(text,all_words):
    words =  re.findall('\p{L}+', text)
    merged = words+ all_words

    return (merged) #set makes unique


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
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
if __name__ == '__main__':
    files = get_files("Selma",'txt')
    all_words = []
    for f in files:
        f_o = open("Selma/" + f, "r")
        content = f_o.read()

        all_words = tokenizeUnique(content, all_words)

    all_words = set(all_words)
    pointing_l={}
    for w in all_words:
        pointing_l[w] = {}
    for f in files:

        f_o = open("Selma/"+f,"r")
        filename = f
        for w in all_words:
            pointing_l[w][filename] = {}
            indexes = ([m.start() for m in re.finditer(w, content)])
            pointing_l[w][filename] = indexes
    print("printing")
    save_obj(pointing_l,"master_index")