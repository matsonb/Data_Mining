import csv
import math
import numpy as np
from collections import defaultdict
import random

total_documents = 0.0
writer_list = ['austen','dickens','shakespeare','et-al']
writers = {}
writer_word_counts = {}
encountered_words = set()
dev_data = {}
dev_data_size = 0.0


"""
Gets document/feature data from txt files, stores in:
author dictionary -> list of documents -> set of words

No return: edits global variables
"""
def parse_files():
    global total_documents, writers, dev_data, encountered_words, dev_data_size
    for writer in writer_list:
        train_size_writer = 0.0
        writers[writer] = []
        dev_data[writer] = []
        file = open(writer+'-parsed.txt')
        csv_file = csv.reader(file)

        row_index = 0.0
        for row in csv_file:
            if row_index % 10 == 0: # add to variable training data
                row_doc = set()
                for word in row:
                    row_doc.add(word)
                dev_data[writer].append(row_doc)
                train_size_writer += 1
                dev_data_size += 1
            else: # add to document training data
                total_documents += 1
                row_doc = set()
                for word in row:
                    row_doc.add(word)
                    encountered_words.add(word)
                writers[writer].append(row_doc)
            row_index += 1
        file.close()


"""
For smoothing: gets the count of all words for each writer
Returns an author dictionary of word dictionaries
"""
def get_word_counts():
    global writer_word_counts
    for writer in writer_list:
        writer_word_counts[writer] = defaultdict(lambda:0)
    for word in encountered_words:
        for writer in writer_list:
            word_count = 0.0
            for doc in writers[writer]:
                if word in doc:
                    word_count += 1
            writer_word_counts[writer][word] = word_count


"""
Performs naive bayes on the given document based on
already calculated counts of words for each author.

Optional param: features specifies which words to use, default is all words read
"""
def naive_bayes(new_document,features = encountered_words):

    probs = [math.log(len(writers[writer])/total_documents) for writer in writer_list]
    
    for word in new_document:
        if not word in features:
            continue
        for i in range(len(writer_list)):
            probs[i] += math.log((writer_word_counts[writer_list[i]][word]+1)/(len(writers[writer_list[i]]) + len(
                features)))
    
    return writer_list[np.argmax(probs)]


"""
Finds a smaller subset of features to use by
finding every single feature that performs the best.
TODO: run this a bunch of times
"""
def naive_feature_select(cutoff):
    good_features = set()
    for word in encountered_words:
        correct = 0.0
        for writer in writer_list:
            for doc in dev_data[writer]:
                if naive_bayes(doc, {word}) == writer:
                    correct += 1
        if correct / dev_data_size > cutoff:
            good_features.add(word)
    
    final_correct = 0.0
    for writer in dev_data:
        for doc in dev_data[writer]:
            if naive_bayes(doc,good_features) == writer:
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
def greedy_feature_select():
    s = [0.0, set()]
    unused_words = encountered_words
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
                    if naive_bayes(doc, t) == writer:
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


def all_features():
    correct = 0.0
    total = 0.0
    for writer in writer_list:
        for doc in dev_data[writer]:
            if naive_bayes(doc) == writer:
                correct += 1
            total += 1
    print(correct)
    print(correct/ dev_data_size)


# defines random mini-batches of the training data
def create_batches(num_batches):
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


def test_set():
    pass


if __name__ == '__main__':
    parse_files()
    get_word_counts()
    print("Finished reading data")
    # naive_feature_select(0.3)
    # naive_feature_select(0.5)
    data_batches = create_batches(5)

