import sys
import math
import regex as re

close_tag = "</s>"
open_tag = "<s>"
def tokenize(text):
    """uses the nonletters to break the text into words
    returns a list of words"""
    # words = re.split('[\s\-,;:!?.’\'«»()–...&‘’“”*—]+', text)
    # words = re.split('[^a-zåàâäæçéèêëîïôöœßùûüÿA-ZÅÀÂÄÆÇÉÈÊËÎÏÔÖŒÙÛÜŸ’\-]+', text)
    # words = re.split('\W+', text)
    words = re.findall('[\S]+', text)

    return words



def makeSentences(text):
    """uses the punctuation and symbols to break the text into words
    returns a list of words"""
    #spaced_tokens = re.sub('([\p{S}\p{P}])', r' \1 ', text)
    #print(spaced_tokens)
    text_without_newlines = re.sub('\n|\t', ' ', text)
    one_sentences = re.findall('[\S\s]*?[\.|\?]', text_without_newlines) #pattern: '\.\s+[A-Z]' eller [A-Z]+[^.]*\.
    new_text = list()
    count = 0
    for s in one_sentences:
        count += 1
        new_text.append(' <s>' + s[:-1].lower() + ' </s>')
    #tokens = one_token_per_line.split()
    #print(new_text)
    #print("count: ", count)
    return new_text #lines


def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency


def count_bigrams(words):
    bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]

    frequency_bigrams = {}
    for bigram in bigrams:
        if bigram in frequency_bigrams:
            frequency_bigrams[bigram] += 1
        else:
            frequency_bigrams[bigram] = 1

    return frequency_bigrams

def unigramModel(words, uniFreq):

    print("Unigram Model")
    print("=====================================================")
    print("w_i       C(w_i)     #words        P(w_i)")  # The unigram model
    print("=====================================================")
    unigramProb = 1
    H = 0
    testSentence = tokenize("det var en gång en katt som hette nils")
    for word in testSentence:
        word = word.lower()
        prob = uniFreq[word] / len(words)
        unigramProb *= prob
        H += math.log(prob, 2)
        print(word, '\t', uniFreq[word], "\t", len(words), "\t", prob)


    prob = uniFreq[close_tag] / len(close_tag)
    unigramProb *= prob

    print(close_tag, '\t', uniFreq[close_tag], "\t", len(close_tag), "\t", prob)
    print("=====================================================")

    geo_prob = math.pow(unigramProb, 1 / (len(testSentence) - 1));
    entropy_rate = -H / (len(testSentence) - 1)
    print("Prob. unigrams: " + str(unigramProb))
    print("Geometric mean prob.: " + str(geo_prob))
    print("Entropy rate: " + str(entropy_rate))
    print("Perplexity: " + str(math.pow(2, entropy_rate)))


def bigramModel(words,uniFreq):

    print("\n\nBigram Model")
    print("=====================================================")

    print("w_i       C(w_i)     #words        P(w_i)")  # The unigram model
    print("=====================================================")

    bigramFreq = count_bigrams(words)

    #Create list of the sentence
    sentence = list()
    sentence.append(open_tag)
    testSentence = tokenize("det var en gång en katt som hette nils")
    for w in testSentence:
        sentence.append(w)
    sentence.append(close_tag)

    #Compute language model

    #set prob to 100%
    bigramProb = 1
    #Entropy
    H = 0

    for i in range(len(sentence)-1):
        #If the next bigram sequence exists, calculate the probability
        if(sentence[i], sentence[i + 1]) in bigramFreq:

            #ci
            wordFreq = uniFreq[sentence[i]]

            #ci_i+1
            nextBigramFreq = bigramFreq[sentence[i],sentence[i+1]]

            prob = nextBigramFreq/wordFreq

            print(sentence[i] + "    " + sentence[i + 1] + "    " + str(nextBigramFreq) + "    " + str(wordFreq) + "    " + str(prob))
        else:
            #fallback to unigram probability.
            prob = uniFreq[sentence[i + 1]] / len(words)
            print(sentence[i] + "     " + sentence[i + 1] + "    " + "0" + "    "
                  + str(uniFreq[sentence[i]]) + "\t" + "0.0 *backoff: " + str(prob))

        bigramProb *= prob
        H += math.log(prob,2)

    geo_prob = math.pow(bigramProb, 1 / (len(sentence) - 1))
    entropy_rate = -H / (len(sentence) - 1)
    print("=====================================================")
    print("Prob. bigrams: " + str(bigramProb))
    print("Geometric mean prob.: " + str(geo_prob))
    print("Entropy rate: " + str(entropy_rate))
    print("Perplexity: " + str(math.pow(2, entropy_rate)))

def get_all_words(parsedtext):
    all_words = list()
    for s in parsedtext:
        temp_words = tokenize(s)

        for t in temp_words:
            if t != '':
                all_words.append(t)
    return all_words

if __name__ == '__main__':
    file = open("Selma2.txt", "r")

    text = file.read()
    parsedtext = makeSentences(text)

    #Check the last five line of the parsing to match with pieres - Its correct.
    #print(parsedtext[-5:])


    all_words = get_all_words(parsedtext)

    uniFreq = count_unigrams(all_words)

    unigramModel(all_words, uniFreq)
   #-----------------------------

    bigramModel(all_words,uniFreq)

