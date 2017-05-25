import csv
from collections import defaultdict
import math
import numpy as np

class naive_bayes_data():
    writer_list = []
    writers = {}
    writer_word_counts = {}
    total_documents = 0.0
    encountered_words = set()

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

        for word in new_document:
            if not word in features:
                continue
            for i in range(len(self.writer_list)):
                smoothed_prob = (self.writer_word_counts[self.writer_list[i]][word] + 1) / (
                len(self.writers[self.writer_list[i]]) + len(features))
                probs[i] += math.log(smoothed_prob)

        return self.writer_list[np.argmax(probs)]

    def set_writers(self, writers):
        self.writers = writers
        self.get_word_counts()
        self.encountered_words = [key for writer in self.writer_list for key in self.writer_word_counts[writer]]
