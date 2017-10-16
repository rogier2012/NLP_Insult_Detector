import nltk
import csv
from nltk.util import ngrams
import textblob.classifiers as classifiers
import math

train_data = []
with open('train.csv', 'r') as csvfile:
    insult_reader = csv.DictReader(csvfile)
    for row in insult_reader:
        train_data.append((row['Comment'], row['Insult']))

# neutral_bow = dict()
# insult_bow = dict()
# for (comment, insult) in train_data:
#     token = nltk.word_tokenize(comment)
#     bigrams = ngrams(token, 1)
#     for bi in bigrams:
#         if insult == '1':
#             insult_bow[bi] = insult_bow.get(bi, 0) + 1
#         else:
#             neutral_bow[bi] = insult_bow.get(bi, 0) + 1
#
# # print(featuresets)
# classifier = nltk.NaiveBayesClassifier.train([(neutral_bow, 0), (insult_bow, 1)])
# print(classifier.show_most_informative_features())

cl = classifiers.NaiveBayesClassifier(train_data)


test_data = []
with open('test_with_solutions.csv', 'r') as csvfile:
    insult_reader = csv.DictReader(csvfile)
    for row in insult_reader:
        test_data.append((row['Comment'], row['Insult']))

# test_sets = [(nltk.tokenize(n), insult) for (n, insult) in train_data]

test_results = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}


for (comment, insult) in test_data:
    label = cl.classify(comment)
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

def print_confusion_matrix(results, positive_label, classifier_name):
    print("Confusion matrix for {}".format(classifier_name))
    print("Positive: {}".format(positive_label))
    print("                    | Actual          | -")
    print("                    | Positive        | Negative")
    print("Predicted Positive  | {:5}           | {:5}".format(results['tp'], results['fp']))
    print("-         Negative  | {:5}           | {:5}".format(results['fn'], results['tn']))
    precision = results['tp'] * 100 / (results['tp'] + results['fp'])
    recall = results['tp'] * 100 / (results['tp'] + results['fn'])
    matthews1 = results['tp'] * results['tn'] - results['fp'] * results['fn']
    matthews2 = (results['tp'] + results['fp']) * (results['tp'] + results['fn']) * (results['tn'] + results['fp']) *\
                (results['tn'] + results['fn'])
    matthews = float(matthews1) * 100 / math.sqrt(matthews2)
    print("Precision    : {:05.2f}%".format(precision))
    print("Recall       : {:05.2f}%".format(recall))
    print("Matthews     : {:05.2f}".format(matthews))

    print("-------------------------------------------------")


print_confusion_matrix(test_results, 'Insult', 'NaiveBayes')
