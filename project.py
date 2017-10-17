import nltk
import csv

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import ensemble
from sklearn import neural_network
import sklearn
import textblob.classifiers
import math

stopWords = set(stopwords.words('english'))
ps = PorterStemmer()
tk = nltk.TweetTokenizer(strip_handles=True, reduce_len=True)

def normalize_comment(comment):
    result = comment.replace('\"', '')
    result = result.lower()
    tokens = nltk.word_tokenize(result)
    wordsFiltered = []

    for w in tokens:
        wordsFiltered.append(ps.stem(w))
    return wordsFiltered

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


def print_confusion_matrix(results, positive_label, classifier_name, n):
    print("-------------------------------------------------")
    print("Confusion matrix for {} using {}-grams".format(classifier_name, n))
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


def extract_features(text,n = 1, extended=False,):
    # make bag of words
    bow = dict()
    for word in nltk.ngrams(text, n):
        if not extended:
            if word not in bow:
                bow[word] = True
        else:
            bow[word] = bow.get(word, 0) + 1
    return bow


class SVMClassifier(textblob.classifiers.NLTKClassifier):
    nltk_class = nltk.SklearnClassifier(sklearn.svm.SVC)


class MyNaiveBayesClassifier(textblob.classifiers.NLTKClassifier):
    nltk_class = nltk.classify.NaiveBayesClassifier


models = [("SVM", nltk.SklearnClassifier(sklearn.svm.LinearSVC())),
          ("NaiveBayes", nltk.classify.NaiveBayesClassifier),
          # ("RandomForest", nltk.SklearnClassifier(ensemble.RandomForestClassifier())),
          # ("MaxEntropy", nltk.classify.MaxentClassifier),
          # ("MLP", nltk.SklearnClassifier(neural_network.MLPClassifier()))
          ]
onlyBayes = [ ("NaiveBayes", nltk.classify.NaiveBayesClassifier)]


def test_classifiers(classifiers, ngrams, extended_bow=False):
    for ndx, (name, cl) in enumerate(classifiers):
        for gram in ngrams:
            test_results = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
            train_features = [(extract_features(text, gram, extended_bow), cls) for text, cls in train_data]

            # print(train_features_list)

            model = cl.train(train_features)
            for (comment, insult) in test_data:
                text_features = extract_features(comment,gram, extended_bow)
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
            print_confusion_matrix(test_results, 'Insult', name, gram)


# test_classifiers(models, [1, 2, 3])
test_classifiers(models, [1, 2, 3, 4, 5, 6, 7], True)
# test_classifiers(onlyBayes, [1, 2, 3])
# test_classifiers(onlyBayes,[1, 2, 3], True)
# for i in range(6):
#     print(extract_features(train_data[i][0]))
#     print(extract_features(train_data[i][0], extended=True))