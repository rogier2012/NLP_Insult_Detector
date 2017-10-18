import csv
import math
import re

# import matplotlib.pyplot as plt
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stopWords = set(stopwords.words('english'))
ps = PorterStemmer()
tk = nltk.TweetTokenizer(strip_handles=True, reduce_len=True)

filter_symbols = ['\\xc2\\xa0', '\\xe2\\x80\\xa6', '\\xe2', '\\xa3', '\\xc2', '\\n', '\\x80', '\\x99', '\\xc3',
                  '\\xb3n',
                  '\\x9d', '\\x9c', '\\xb4', '\\xf0', '\\x9f']


def stemming(comment):
    wordsFiltered = []

    for w in comment:
        wordsFiltered.append(ps.stem(w))
    return wordsFiltered

def normalize_comment(comment):
    result = comment.replace('\"', '')
    result = result.lower()

    result = re.sub(r"\\x[a-f0-9]{2}", '', result)
    result = re.sub(r"\\n", '', result)
    result = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", 'URL', result)
    result = re.sub(r"@[A-Za-z0-9]+", '@', result)
    tokens = nltk.word_tokenize(result)
    return stemming(tokens)


train_data = []
with open('train.csv', 'r', encoding='unicode-escape') as csvfile:
    insult_reader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
    for row in insult_reader:
        com = normalize_comment(row['Comment'])
        # print(com.encode('utf-8').decode('unicode-escape'))
        train_data.append((com, row['Insult']))

# def histogram():
#     histo = [x for (c, x) in train_data]
#     plt.hist(histo, bins=2, rwidth=0.8, label="Insult data set")
#     plt.xlabel('Data class')
#     plt.ylabel('Number of instances')
#     plt.savefig('histogram.png')


test_data = []

with open('test_with_solutions.csv', 'r', encoding='unicode-escape') as csvfile:
    insult_reader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
    for row in insult_reader:
        com = normalize_comment(row['Comment'])
        test_data.append((com, row['Insult']))


def print_confusion_matrix(results, positive_label, classifier_name, n, cutoff, counter, extenden_bow):
    print("-------------------------------------------------")
    print(
        "{}. Confusion matrix for {} using {}-grams with extended bow:{} and {} cutoff".format(counter, classifier_name,
                                                                                               n, extenden_bow, cutoff))
    print("Positive: {}".format(positive_label))
    print("                    | Actual          | -")
    print("                    | Positive        | Negative")
    print("Predicted Positive  | {:5}           | {:5}".format(results['tp'], results['fp']))
    print("-         Negative  | {:5}           | {:5}".format(results['fn'], results['tn']))

    if (results['tp'] + results['fp'] + results['tn'] + results['fn']) > 0:
        accuracy = (results['tp'] + results['tn']) * 100 / (
            results['tp'] + results['fp'] + results['tn'] + results['fn'])
    else:
        accuracy = 0
    if (results['tp'] + results['fp']) > 0:
        precision = results['tp'] * 100 / (results['tp'] + results['fp'])
    else:
        precision = 100
    if (results['tp'] + results['fn']) > 0:
        recall = results['tp'] * 100 / (results['tp'] + results['fn'])
    else:
        recall = 0

    pxr = precision * recall

    f3 = (1 + 9) * pxr / ((9 * precision) + recall)
    f03 = (1 + (1 / 9)) * pxr / (((1 / 9) * precision) + recall)

    matthews1 = results['tp'] * results['tn'] - results['fp'] * results['fn']
    matthews2 = (results['tp'] + results['fp']) * (results['tp'] + results['fn']) * (results['tn'] + results['fp']) * \
                (results['tn'] + results['fn'])
    if matthews2 > 0:
        matthews = float(matthews1) * 100 / math.sqrt(matthews2)
    else:
        matthews = 0
    print("Accuracy     : {:6.2f}%".format(accuracy))
    print("Precision    : {:6.2f}%".format(precision))
    print("Recall       : {:6.2f}%".format(recall))
    print("F-0.33 Score : {:6.2f}%".format(f03))
    print("F-3 Score    : {:6.2f}%".format(f3))
    print("Matthews     : {:6.2f}%".format(matthews))
    print(
        "& {:6.2f}\\% & {:6.2f}\\% & {:6.2f}\\% & {:6.2f}\\% & {:6.2f}\\%".format(precision, recall, f03, f3, matthews))
    index = "{} ({}, {}, {})".format(classifier_name, extenden_bow, n, cutoff)
    return (index, (precision, recall, f03, f3, matthews))


def extract_features(text, n=1, extended=0, ):
    # make bag of words
    bow = dict()
    for word in nltk.ngrams(text, n):
        if extended == 0:
            if word not in bow:
                bow[word] = True
        else:
            bow[word] = bow.get(word, 0) + 1
    return bow


models = [("SVM", nltk.SklearnClassifier(sklearn.svm.SVC(probability=True))),
          ("NaiveBayes", nltk.classify.NaiveBayesClassifier),
          # ("RandomForest", nltk.SklearnClassifier(ensemble.RandomForestClassifier())),
          # ("MaxEntropy", nltk.classify.MaxentClassifier),
          # ("MLP", nltk.SklearnClassifier(neural_network.MLPClassifier()))
          ]
onlyBayes = [("NaiveBayes", nltk.classify.NaiveBayesClassifier)]


def test_classifiers(classifiers, ngrams, cutoffs):
    test_counter = 0
    scores = dict()
    for i in range(2):
        for ndx, (name, cl) in enumerate(classifiers):
            for gram in ngrams:
                test_results = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
                train_features = [(extract_features(text, gram, i), cls) for text, cls in train_data]

                # print(train_features_list)

                model = cl.train(train_features)
                for cutoff in cutoffs:
                    for (comment, insult) in test_data:
                        text_features = extract_features(comment, gram, i)
                        dist = model.prob_classify(text_features)
                        # label = model.classify(text_features)
                        if insult == '1':
                            if dist.prob('1') > cutoff:
                                test_results['tp'] = test_results.get('tp', 0) + 1
                            else:
                                # print(dist.prob('1'))
                                test_results['fn'] = test_results.get('fn', 0) + 1
                        elif insult == '0':
                            if dist.prob('1') > cutoff:
                                test_results['fp'] = test_results.get('fp', 0) + 1
                            else:
                                test_results['tn'] = test_results.get('tn', 0) + 1
                    test_counter = test_counter + 1
                    idx, score = print_confusion_matrix(test_results, 'Insult', name, gram, cutoff, test_counter, i)
                    scores[idx] = score
    measures = ['Precision', 'Recall', 'F1/3', 'F3', 'MCC']
    for ndx, measure in enumerate(measures):
        highest = ""
        highest_score = 0
        for k, v in scores.items():
            if v[ndx] > highest_score:
                highest = k
                highest_score = v[ndx]
        print("Highest {}: {} with {}".format(measure, highest, highest_score))


test_classifiers(models, [1, 2], [0.5, 0.6, 0.7, 0.8, 0.9])
# test_classifiers(models, [1, 2], [0.3, 0.6, 0.7,0.8, 0.9, 0.95], True)
# test_classifiers(onlyBayes, [1, 2, 3])
# test_classifiers(onlyBayes,[1, 2, 3], True)
# for i in range(6):
#     print(train_data[i][0])
# print(extract_features(train_data[i][0], extended=True))
