import csv
import math
import numpy as np
from collections import defaultdict
import random

#total_documents = 0.0
writer_list = ['austen','dickens','shakespeare','et-al']
# writers = {}
# writer_word_counts = {}
# encountered_words = set()
# dev_data = {}
# dev_data_size = 0.0


"""
Gets document/feature data from txt files, stores in:
author dictionary -> list of documents -> set of words

No return: edits global variables
"""
def parse_files():
    total_documents = 0.0
    writers = {}
    for writer in writer_list:
        train_size_writer = 0.0
        writers[writer] = []
        file = open(writer+'-parsed.txt')
        csv_file = csv.reader(file)

        row_index = 0.0
        for row in csv_file:
            # if row_index % 10 == 0: # add to variable training data
            #     row_doc = set()
            #     for word in row:
            #         row_doc.add(word)
            #     dev_data[writer].append(row_doc)
            #     train_size_writer += 1
            #     dev_data_size += 1
            # else: # add to document training data
            total_documents += 1
            row_doc = set()
            for word in row:
                row_doc.add(word)
            writers[writer].append(row_doc)
            row_index += 1
        file.close()
    return total_documents, writers

"""
Randomly removes 10% of the data for purposes of testing parameters and features
"""
def split_10_data(full_data):
    data_10_per = {}
    data_90_per = {}
    for writer in writer_list:
        doc_list = list(full_data[writer])
        docs_10_per = [doc_list.pop(random.randrange(len(doc_list))) for i in range(int(len(doc_list)/10.0))]
        data_10_per[writer] = docs_10_per
        data_90_per[writer] = doc_list
    return data_90_per, data_10_per


"""
For smoothing: gets the count of all words for each writer
Returns an author dictionary of word dictionaries
"""
def get_word_counts(train_data):
    train_data_counts = {}
    for writer in writer_list:
        train_data_counts[writer] = defaultdict(lambda:0.0)
    for writer in writer_list:
        for doc in train_data[writer]:
            for word in doc:
                train_data_counts[writer][word] += 1
    return train_data_counts


"""
Performs naive bayes on the given document based on
already calculated counts of words for each author.

Optional param: features specifies which words to use, default is all words read
"""
def naive_bayes(new_document, features, train_data, train_data_counts, total_documents):

    probs = [math.log(len(train_data[writer])/total_documents) for writer in writer_list]
    
    for word in new_document:
        if not word in features:
            continue
        for i in range(len(writer_list)):
            smoothed_prob = (train_data_counts[writer_list[i]][word]+1)/(len(train_data[writer_list[i]])+len(features))
            probs[i] += math.log(smoothed_prob)
    
    return writer_list[np.argmax(probs)]


"""
Finds a smaller subset of features to use by
finding every single feature that performs the best.
TODO: run this a bunch of times
"""
def naive_feature_select(cutoff, possible_features, train_data, train_data_counts, dev_data, total_documents):
    dev_data_size = sum([len(values) for values in dev_data.itervalues])
    good_features = set()
    for word in possible_features:
        correct = 0.0
        for writer in writer_list:
            for doc in dev_data[writer]:
                if naive_bayes(doc, {word}, train_data, train_data_counts, total_documents) == writer:
                    correct += 1
        if correct / dev_data_size > cutoff:
            good_features.add(word)
    
    final_correct = 0.0
    for writer in dev_data:
        for doc in dev_data[writer]:
            if naive_bayes(doc, good_features, train_data, train_data_counts, total_documents) == writer:
                final_correct += 1

    print(good_features)
    print(len(good_features))
    print(final_correct / dev_data_size)


"""
Finds a smaller subset of features to use by
iteratively choosing next feature to add on by choosing the one that improves
the current set the most

Stops when the the addition of any remaining feature would decrease the accuracy
"""
def greedy_feature_select(possible_features, train_data, train_data_counts, dev_data, total_documents):
    s = [0.0, set()]
    unused_words = possible_features
    while True:
        print("")
        print("")
        print("")
        best_t = [0,set(), '']
        for word in unused_words:
            t = s[1].union({word})
            t_score = 0.0
            for writer in writer_list:
                for doc in dev_data[writer]:
                    if naive_bayes(doc, t, train_data, train_data_counts, total_documents) == writer:
                        t_score += 1
            if t_score > best_t[0]:
                best_t = [t_score, t, word]
        if best_t[0] >= s[0]:
            s = best_t[:2]
            unused_words.remove(best_t[2])
            print(s)
        else:
            break
    return s


def all_features(features, train_data, train_data_counts, dev_data, total_documents):
    correct = 0.0
    total = 0.0
    for writer in writer_list:
        for doc in dev_data[writer]:
            if naive_bayes(doc, features, train_data, train_data_counts, total_documents) == writer:
                correct += 1
            total += 1
    return correct


# defines random mini-batches of the training data
def create_batches(num_batches, writers):
    batches = []
    for i in range(num_batches):
        batches.append({})
        for writer in writer_list:
            batches[i][writer] = []
    for writer in writer_list:
        author_data = list(writers[writer])
        random.shuffle(author_data)
        batch_size = int(math.ceil(float(len(author_data))/num_batches))
        mini_batches = [author_data[i:i + batch_size] for i in range(0, len(author_data), batch_size)]
        for i in range(len(mini_batches)):
            batches[i][writer] = mini_batches[i]
    return batches


def cross_validation(num_batches, features, writers, total_documents):
    batches = create_batches(num_batches, writers)
    correct = 0.0
    total = 0.0
    for test in range(num_batches):
        train_data = {}
        for writer in writer_list:
            train_data[writer] = []
        test_data = batches[test]
        for index in range(len(batches)):
            if index != test:
                for writer in batches[index]:
                    train_data[writer] += batches[index][writer]
        train_data_counts = get_word_counts(train_data)

        for writer in test_data:
            for doc in test_data[writer]:
                label = naive_bayes(doc, features, train_data, train_data_counts, total_documents)
                total += 1
                if label == writer:
                    correct += 1
    return correct/total

def main():
    total_documents, writers = parse_files()
    writers, dev_data = split_10_data(writers)
    writers_word_counts = get_word_counts(writers)
    encountered_words = [key for writer in writer_list for key in writers_word_counts[writer]]
    print("Finished reading data")
    feature_words = ['hath', 'example', 'feelings', 'manners', 'thy', 'allow', 'b', 'dorrit', 'friendship',
                     'carriage', 'cant', 'plan', 'handed', 'conduct', 'candle', 'her', 'crows', 'repeated',
                     'was', 'faded', 'thin', 'excite']
    print cross_validation(5, feature_words, writers, total_documents)

if __name__ == '__main__':
    main()


    # feature_words = ['hath', 'example', 'feelings', 'manners', 'thy', 'allow', 'b', 'dorrit', 'friendship',
    #                  'carriage', 'cant', 'plan', 'handed', 'conduct', 'candle', 'her', 'crows', 'repeated',
    #                  'was', 'faded', 'thin', 'excite']
    # print cross_validation(5, feature_words)

