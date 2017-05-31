'''
A class to store the document data
'''



import csv
from collections import defaultdict
import math

class naive_bayes_data():
    # The writers
    writer_list = []
    # dicctionary, writer : list of documents
    writers = {}
    # 2d dictionary, writer : word : number of documents by writer containing word
    writer_word_counts = {}
    # Total number of documents (so we don't need to loop through writer_list to access this)
    total_documents = 0.0
    # Holds all encountered words
    encountered_words = set()


    # constructor
    def __init__(self, writer_list):
        self.writer_list = writer_list
        self.parse_files()

    def parse_files(self):
        """
        Gets document/feature data from txt files, stores in:
        author dictionary -> list of documents -> set of words
        """
        for writer in self.writer_list:
            train_size_writer = 0.0
            self.writers[writer] = []
            file = open(writer + '-parsed.txt')
            csv_file = csv.reader(file)

            row_index = 0.0
            for row in csv_file:

                self.total_documents += 1
                row_doc = set()
                for word in row:
                    row_doc.add(word)
                    self.encountered_words.add(word)
                self.writers[writer].append(row_doc)
                row_index += 1
            file.close()

    def get_word_counts(self):
        """
        For smoothing: gets the count of all words for each writer
        Returns an author dictionary of word dictionaries
        """
        for writer in self.writer_list:
            self.writer_word_counts[writer] = defaultdict(lambda: 0.0)
        for writer in self.writer_list:
            for doc in self.writers[writer]:
                for word in doc:
                    self.writer_word_counts[writer][word] += 1

    def naive_bayes(self, new_document, features = encountered_words):
        """
        Performs naive bayes on the given document based on
        already calculated counts of words for each author.

        Optional param: features specifies which words to use, default is all words read
        """
        probs = [math.log(len(self.writers[writer]) / self.total_documents) for writer in self.writer_list]

        for word in features:
            if not word in new_document:
                continue
            for i in range(len(self.writer_list)):
                smoothed_prob = (self.writer_word_counts[self.writer_list[i]][word] + 1) / (
                len(self.writers[self.writer_list[i]]) + len(features))
                probs[i] += math.log(smoothed_prob)

        # This gets argmax(probs) but empirically this way seems faster
        max_prob = max(probs)
        for i in range(len(probs)):
            if probs[i] == max_prob:
                return self.writer_list[i]

    # Specify a subset of the documents to be the training set
    def set_writers(self, writers):
        self.writers = writers
        self.get_word_counts()
        self.encountered_words = [key for writer in self.writer_list for key in self.writer_word_counts[writer]]
