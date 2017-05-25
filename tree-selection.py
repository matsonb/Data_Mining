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


def expected_information(sample_sizes):
    exp_inf = 0.0
    total_sample_size = sum(sample_sizes) * 1.0
    if total_sample_size == 0:
        return 0
    probs = [sample_sizes[i]/total_sample_size for i in range(len(sample_sizes))]
    for prob in probs:
        if prob != 0:
            exp_inf += prob * math.log(prob,2)
    return -1 * exp_inf
    


def entropy(word,sample):
    has_word = []
    no_has_word = []
    for author in sample:
        has_word.append(0.0)
        no_has_word.append(0.0)
        for doc in sample[author]:
            if word in doc:
                has_word[-1] += 1
            else:
                no_has_word[-1] += 1
    ent = 0.0
    total_docs = sum(has_word) + sum(no_has_word)
    ent += sum(has_word)/total_docs * expected_information(has_word)
    ent += sum(no_has_word)/total_docs * expected_information(no_has_word)
    return ent

def information_gain(word,sample):
    sample_sizes = [len(sample[author]) for author in sample]
    return expected_information(sample_sizes) - entropy(word,sample)


def split_information(word,sample):
    has_word = []
    no_has_word = []
    for author in sample:
        has_word.append(0.0)
        no_has_word.append(0.0)
        for doc in sample[author]:
            if word in doc:
                has_word[-1] += 1
            else:
                no_has_word[-1] += 1
    total_docs = sum(has_word) + sum(no_has_word)
    split = 0.0
    if sum(has_word) != 0:
        split += sum(has_word)/total_docs * math.log(sum(has_word)/total_docs,2)
    if sum(no_has_word) != 0:
        split += sum(no_has_word)/total_docs * math.log(sum(no_has_word)/total_docs,2)
    return -1 * split

def gain_ratio(word,sample):
    split_info = split_information(word,sample)
    info_gain = information_gain(word,sample)
    if split_info == 0:
        return 0
    return info_gain/split_info


def c45(sample,depth,split_words):
    best_ratio = (0.0,'')
    for word in encountered_words:
        if word in split_words:
            continue
        word_ratio = gain_ratio(word,sample)
        if word_ratio > best_ratio[0]:
            best_ratio = (word_ratio,word)
    if best_ratio[0] == 0:
        return []
    attr = best_ratio[1]
    if depth < 2:
        has_attr = {}
        no_has_attr = {}
        has_attr_length = 0.0
        no_has_attr_length = 0.0
        for author in sample:
            has_attr[author] = []
            no_has_attr[author] = []
            for doc in sample[author]:
                if attr in doc:
                    has_attr[author].append(doc)
                    has_attr_length += 1
                else:
                    no_has_attr[author].append(doc)
                    no_has_attr_length += 1
        left_words = []
        right_words = []
        new_split_words = split_words + [attr]
        if has_attr_length > 0:
            left_words = c45(has_attr,depth + 1,new_split_words)
        if no_has_attr_length > 0:
            right_words = c45(no_has_attr,depth + 1,new_split_words)
        return [attr] + left_words + right_words
    else:
        return [attr]



def main():
    parse_files()
    print(c45(dev_data,0,[]))


if __name__ == '__main__':
    main()