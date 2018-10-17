
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.externals import joblib

from pathlib import Path

import transition
import conll

feature_names_1 = [
    'stack0_POS',
    'stack0_word',
    'queue0_POS',
    'queue0_word',
    'can-re',
    'can-la'
]

feature_names_2 = [
    'stack0_POS',
    'stack1_POS',
    'stack0_word',
    'stack1_word',
    'queue0_POS',
    'queue1_POS',
    'queue0_word',
    'queue1_word',
    'can-re',
    'can-la'
]


feature_names_3 = [
    'stack0_POS',
    'stack1_POS',
    'stack0_word',
    'stack1_word',
    'queue0_POS',
    'queue1_POS',
    'queue0_word',
    'queue1_word',
    'next_word_POS',
    'next_word',
    'prev_word_POS',
    'prev_word',
    'can-re',
    'can-la']

FEATURE_NAMES = feature_names_3
def extract(stack, queue, graph, feature_names, sentence):
    features = []
    features.append(stack[0]["form"])
    features.append(stack[0]["postag"])
    features.append(queue[0]["form"])
    features.append(queue[0]["postag"])
    return features
def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

def extract(stack, queue, graph, feature_names, sentence):
    features = {}
    for fn in feature_names:
        if fn == 'stack0_POS':
            if stack:
                features["stack0_POS"] = stack[0]["postag"]
            else:
                features["stack0_POS"] = "nil"
        if fn == 'stack1_POS':
            if len(stack) > 1:
                features["stack1_POS"] = stack[1]["postag"]
            else:
                features["stack1_POS"] = "nil"
        if fn == 'stack0_word':

            if stack:
                features["stack0_word"] = stack[0]["form"]
            else:
                features["stack0_word"] = "nil"
        if fn == 'stack1_word':
            if stack and len(stack) > 1:
                features["stack1_word"] = stack[1]["form"]
            else:
                features["stack1_word"] = "nil"
        if fn == 'queue0_POS':
            if queue:
                features["queue0_POS"] = queue[0]["postag"]
            else:
                features["queue0_POS"] = "nil"
        if fn == 'queue1_POS':
            if queue and len(queue) > 1:
                features["queue1_POS"] = queue[1]["postag"]
            else:
                features["queue1_POS"] = "nil"
        if fn == 'queue0_word':
            if queue:
                features["queue0_word"] = queue[0]["form"]
            else:
                features["queue0_word"] = "nil"
        if fn == 'queue1_word':
            if queue and len(queue) > 1:
                features["queue1_word"] = queue[1]["form"]
            else:
                features["queue1_word"] = "nil"
        if fn == 'can-re':
            features["can-re"] = str(transition.can_reduce(stack, graph))

        if fn == 'can-la':
            features["can-la"] = str(transition.can_leftarc(stack, graph))

        if fn == 'next_word_POS':
            features["next_word_POS"] = "nil"
            if int(queue[0]["id"]) < len(sentence) - 1:
                #Next sentece +1
                w = sentence[int(queue[0]["id"]) + 1]
                features["next_word_POS"] = w['postag']
        if fn == 'next_word':
            features["next_word"] = "nil"
            if int(queue[0]["id"]) < len(sentence) - 1:
                #Next sentece +1
                w = sentence[int(queue[0]["id"]) + 1]
                features["next_word"] = w['form']

        if fn == 'prev_word_POS':
            features["prev_word_POS"] = "nil"
            if int(queue[0]["id"]) < len(sentence) - 1:
                # prev sentece -1
                w = sentence[int(queue[0]["id"]) - 1]
                features["prev_word_POS"] = w['postag']
        if fn == 'prev_word':
            features["prev_word"] = "nil"
            if int(queue[0]["id"]) < len(sentence) - 1:
                # prev sentece -1
                w = sentence[int(queue[0]["id"]) - 1]
                features["prev_word"] = w['form']

    return features

def computeVectors(file):
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    sentences = conll.read_sentences(file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    X = []
    y = []

    for sentence in formatted_corpus:

        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []
        while queue:
            features = extract(stack, queue, graph, FEATURE_NAMES, sentence)
            stack, queue, state, trans = reference(stack, queue, graph)
            transitions.append(trans)
            X.append(features)
            y.append(trans)
    return X, y

def computeVectorsModel(file, model, v, le):
        column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

        sentences = conll.read_sentences(file)
        formatted_corpus = conll.split_rows(sentences, column_names_2006)

        X = []
        y = []

        for sentence in formatted_corpus:

            stack = []
            queue = list(sentence)
            graph = {}
            graph['heads'] = {}
            graph['heads']['0'] = '0'
            graph['deprels'] = {}
            graph['deprels']['0'] = 'ROOT'

            while queue:
                features = extract(stack, queue, graph, FEATURE_NAMES, sentence)
                features_encoded = v.transform(features)
                trans_nr = model.predict(features_encoded)[0]
                trans = le.inverse_transform(trans_nr)
                stack, queue, graph, trans = parse_ml(stack, queue, graph, trans)
                X.append(features)
                y.append(trans)

            stack, graph = transition.empty_stack(stack, graph)

            for word in sentence:
                word_id = word['id']
                try:
                    word['head'] = graph['heads'][word_id]
                    word['phead'] = graph['heads'][word_id]
                except KeyError:
                    word['head'] = '_'
                    word['phead'] = '_'

                try:
                    word['deprel'] = graph['deprels'][word_id]
                    word['pdeprel'] = graph['deprels'][word_id]
                except KeyError:
                    word['deprel'] = '_'
                    word['pdeprel'] = '_'

        conll.save('results.txt', formatted_corpus, column_names_2006)
        return X, y
def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra' and transition.can_rightarc(stack):
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    if stack and trans[:2] == 'la' and transition.can_leftarc(stack, graph):
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'
    if stack and trans == 're' and transition.can_reduce(stack,graph):
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

if __name__ == '__main__':
    train_file = '../swedish_talbanken05_train.conll.txt'
    test_file = '../swedish_talbanken05_test.conll.txt'

    X_u, y_u = computeVectors(train_file)
    X_test_u, y_test_u = computeVectors(test_file)

    file = Path("dictvec.pkl")
    if file.is_file():
        print("Loading model..")
        v = joblib.load('dictvec.pkl')
    else:
        v = DictVectorizer(sparse=True)
        joblib.dump(v, 'dictvec.pkl')



    file = Path("labelencoder.pkl")
    if file.is_file():
        print("Loading model..")
        le = joblib.load('labelencoder.pkl')
    else:
        le = preprocessing.LabelEncoder()
        joblib.dump(le, 'labelencoder.pkl')
    print("Transforming vectors")


    y = le.fit_transform(y_u)

    X = v.fit_transform(X_u)


    model = 0


    file = Path("classifier.pkl")
    if file.is_file():
        print("Loading model..")
        model = joblib.load('classifier.pkl')
    else:
        print("Training model..")
        classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
        model = classifier.fit(X, y)
        print("Saving classifier..")
        joblib.dump(classifier, 'classifier.pkl')



    #print("Predicting training data...")
    #y_predicted = model.predict(X)

    #training_acc = accuracy_score(y, y_predicted)
    #print("Accuracy (Training) :" + str(training_acc))

    #test
    #y_U = le.transform(y_test_u)
    #X_U = v.transform(X_test_u)
    #y_pred_U = model.predict(X_U)
    #aS_test = accuracy_score(y_U,y_pred_U)
    #print("Accuracy (Test) :" + str(aS_test))

    #Predict with computed model
    X_test_U, y_test_U = computeVectorsModel(test_file, model, v, le)