import nltk
import csv
from sklearn import ensemble
from sklearn import neural_network
import sklearn
import textblob.classifiers
import math


def normalize_comment(comment):
    result = comment.replace('\"', '')
    result = result.lower()
    return result

train_data = []
with open('train.csv', 'r', encoding='unicode-escape') as csvfile:
    insult_reader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
    for row in insult_reader:
        com = normalize_comment(row['Comment'])
        # print(com.encode('utf-8').decode('unicode-escape'))
        train_data.append((com, row['Insult']))

test_data = []

with open('test_with_solutions.csv', 'r', encoding='unicode-escape') as csvfile:
    insult_reader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
    for row in insult_reader:
        com = normalize_comment(row['Comment'])
        test_data.append((com, row['Insult']))


def print_confusion_matrix(results, positive_label, classifier_name):
    print("-------------------------------------------------")
    print("Confusion matrix for {}".format(classifier_name))
    print("Positive: {}".format(positive_label))
    print("                    | Actual          | -")
    print("                    | Positive        | Negative")
    print("Predicted Positive  | {:5}           | {:5}".format(results['tp'], results['fp']))
    print("-         Negative  | {:5}           | {:5}".format(results['fn'], results['tn']))
    if (results['tp'] + results['fp']) > 0:
        precision = results['tp'] * 100 / (results['tp'] + results['fp'])
    else:
        precision = 100
    if (results['tp'] + results['fn']) > 0:
        recall = results['tp'] * 100 / (results['tp'] + results['fn'])
    else:
        recall = 0
    matthews1 = results['tp'] * results['tn'] - results['fp'] * results['fn']
    matthews2 = (results['tp'] + results['fp']) * (results['tp'] + results['fn']) * (results['tn'] + results['fp']) * \
                (results['tn'] + results['fn'])
    if matthews2 > 0:
        matthews = float(matthews1) * 100 / math.sqrt(matthews2)
    else:
        matthews = 0
    print("Precision    : {:6.2f}%".format(precision))
    print("Recall       : {:6.2f}%".format(recall))
    print("Matthews     : {:6.2f}%".format(matthews))


def extract_features(text, extended=False):
    # tokenize
    tokens = nltk.word_tokenize(text)
    # make bag of words
    bow = dict()

    for gram in nltk.ngrams(tokens,2):
        if not extended:
            if gram not in bow:
                bow[gram] =  True
        else:
            bow[gram] = bow.get(gram, 0) + 1
    return bow


class SVMClassifier(textblob.classifiers.NLTKClassifier):
    nltk_class = nltk.SklearnClassifier(sklearn.svm.SVC)


class MyNaiveBayesClassifier(textblob.classifiers.NLTKClassifier):
    nltk_class = nltk.classify.NaiveBayesClassifier


models = [("SVM", nltk.SklearnClassifier(sklearn.svm.LinearSVC())),
          ("NaiveBayes", nltk.classify.NaiveBayesClassifier),
          ("RandomForest", nltk.SklearnClassifier(ensemble.RandomForestClassifier())),
          # ("MaxEntropy", nltk.classify.MaxentClassifier),
          ("MLP", nltk.SklearnClassifier(neural_network.MLPClassifier()))
          ]
onlyBayes = [ ("NaiveBayes", nltk.classify.NaiveBayesClassifier)]


def test_classifiers(classifiers, extended_bow=False):
    for ndx, (name, cl) in enumerate(classifiers):
        test_results = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

        train_features = [(extract_features(d, extended_bow), c) for d, c in train_data]
        model = cl.train(train_features)
        for (comment, insult) in test_data:
            text_features = extract_features(comment, extended_bow)
            label = model.classify(text_features)
            if insult == '1':
                if label == '1':
                    test_results['tp'] = test_results.get('tp', 0) + 1
                elif label == '0':
                    test_results['fn'] = test_results.get('fn', 0) + 1
            elif insult == '0':
                if label == '1':
                    test_results['fp'] = test_results.get('fp', 0) + 1
                elif label == '0':
                    test_results['tn'] = test_results.get('tn', 0) + 1
        print_confusion_matrix(test_results, 'Insult', name)


test_classifiers(models)
test_classifiers(models, True)
# test_classifiers(onlyBayes)
# test_classifiers(onlyBayes, True)
# for i in range(2):
#     print(extract_features(train_data[i][0]))
#     print(extract_features(train_data[i][0], True))