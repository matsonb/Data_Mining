import csv
import math
import numpy as np
from collections import defaultdict
import random

total_documents = 0
writer_list = ['austen','dickens','shakespeare','et-al']
writers = {}
encountered_words = set()
dev_data = {}
dev_data_size = 0

def parse_files():
    global total_documents, writers, dev_data, encountered_words, dev_data_size
    for writer in writer_list:
        train_size_writer = 0
        writers[writer] = []
        dev_data[writer] = []
        file = open(writer+'-parsed.txt')
        csv_file = csv.reader(file)

        row_index = 0
        for row in csv_file:
            if row_index % 70 == 0: # add to variable training data
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
        print(writer + ' ' + str(train_size_writer))
        file.close()       

def get_word_counts():
    writer_word_counts = {}
    for writer in writer_list:
        writer_word_counts[writer] = defaultdict(lambda:0)
    for word in encountered_words:
        for writer in writer_list:
            word_count = 0
            for doc in writers[writer]:
                if word in doc:
                    word_count += 1
            writer_word_counts[writer][word] = word_count
    return writer_word_counts
    
def naive_bayes(writer_word_counts,new_document,features = encountered_words):

    probs = [math.log(len(writers[writer])/total_documents) for writer in writer_list]
    
    for word in new_document:
        if not word in features:
            continue
        for i in range(len(writer_list)):
            probs[i] += math.log((writer_word_counts[writer_list[i]][word]+1)/(len(writers[writer_list[i]]) + len(encountered_words)))
    
    return writer_list[np.argmax(probs)]
    
        
def main():
    parse_files()
    exit()
    writer_word_counts = get_word_counts()
    good_features = set()
    print(dev_data_size)
    for word in encountered_words:
        correct = 0.0
        for writer in writer_list:
            for doc in dev_data[writer]:
                if naive_bayes(writer_word_counts,doc, {word}) == writer:
                    correct += 1
        if correct / dev_data_size > .5:
            good_features.add(word)
    print(good_features)
    print(len(good_features))
    
    final_correct = 0.0
    for writer in dev_data:
        for doc in dev_data[writer]:
            if naive_bayes(writer_word_counts,doc,good_features) == writer:
                final_correct += 1
    print(final_correct / dev_data_size)
    
    all_correct = 0.0
    for writer in dev_data:
        for doc in dev_data[writer]:
            if naive_bayes(writer_word_counts,doc) == writer:
                all_correct += 1
    print(all_correct / dev_data_size)
        # take ~10 sample documents from each author
        # run classifier on those documents
        # keep track of correct ones
        # if more than .3 correct, add to good_features
        
    
if __name__ == '__main__':
    main()
