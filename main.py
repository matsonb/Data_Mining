import math
import random
import naive_bayes_data
import sys

writer_list = ['austen', 'dickens', 'shakespeare', 'et-al']
cutoffs = [.43, .435,.44]


def split_10_data(full_data):
    """
    Randomly removes 10% of the data for purposes of testing parameters and features
    """
    data_10_per = {}
    data_90_per = {}
    for writer in writer_list:
        doc_list = list(full_data[writer])
        docs_10_per = [doc_list.pop(random.randrange(len(doc_list))) for i in range(int(len(doc_list) / 10.0))]
        data_10_per[writer] = docs_10_per
        data_90_per[writer] = doc_list
    return data_90_per, data_10_per


def naive_feature_select(cutoff, data_holder, dev_data):
    """
    Finds a smaller subset of features to use by
    finding every single feature that performs the best.
    """
    dev_data_size = sum([len(dev_data[author]) for author in dev_data])
    good_features = set()
    # Go through every word, and add to good_features if it does better than cutoff
    for word in data_holder.encountered_words:
        correct = 0.0
        for writer in writer_list:
            for doc in dev_data[writer]:
                if data_holder.naive_bayes(doc, {word}) == writer:
                    correct += 1
        if correct / dev_data_size > cutoff:
            good_features.add(word)
    return good_features


def greedy_feature_select(data_holder, dev_data, max_iters):
    """
    Finds a smaller subset of features to use by
    iteratively choosing next feature to add on by choosing the one that improves
    the current set the most

    Stops when the the addition of any remaining feature would decrease the accuracy
    or when we have done max_iters iterations
    """
    s = [0.0, set()]
    unused_words = data_holder.encountered_words
    while len(s[1]) < max_iters:
        best_t = [0, set(), '']
        # Add each word in turn to s[1]
        for word in unused_words:
            t = s[1].union({word})
            t_score = 0.0
            for writer in writer_list:
                for doc in dev_data[writer]:
                    if data_holder.naive_bayes(doc, t) == writer:
                        t_score += 1
            # keep the best possible superset
            if t_score > best_t[0]:
                best_t = [t_score, t, word]
        # while we keep improving, add that word to s[1] and repeat
        if best_t[0] >= s[0]:
            s = best_t[:2]
            unused_words.remove(best_t[2])
            print len(s[1]) # this takes a while, so give an update each iteration
        else:
            break
    return s[1]


def create_batches(num_batches, writers):
    """ defines random mini-batches of the training data, used in cross_validation """
    batches = []
    for i in range(num_batches):
        batches.append({})
        for writer in writer_list:
            batches[i][writer] = []
    for writer in writer_list:
        author_data = list(writers[writer])
        random.shuffle(author_data)
        batch_size = int(math.ceil(float(len(author_data)) / num_batches))
        mini_batches = [author_data[i:i + batch_size] for i in range(0, len(author_data), batch_size)]
        for i in range(len(mini_batches)):
            batches[i][writer] = mini_batches[i]
    return batches


def cross_validation(num_batches, features, data_holder):
    """ runs cross validation naive bayes on the data with the given features and number of batches """
    features = set(features)
    batches = create_batches(num_batches, data_holder.writers)
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
        data_holder.set_writers(train_data)

        for writer in test_data:
            for doc in test_data[writer]:
                label = data_holder.naive_bayes(doc, features)
                total += 1
                if label == writer:
                    correct += 1
    return correct / total


def expected_information(sample_sizes):
    # Used to calculate entropy
    exp_inf = 0.0
    total_sample_size = sum(sample_sizes) * 1.0
    if total_sample_size == 0:
        return 0
    probs = [sample_sizes[i] / total_sample_size for i in range(len(sample_sizes))]
    for prob in probs:
        if prob != 0:
            exp_inf += prob * math.log(prob, 2)
    return -1 * exp_inf


def entropy(word, sample):
    # used to calculate information gain
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
    ent += sum(has_word) / total_docs * expected_information(has_word)
    ent += sum(no_has_word) / total_docs * expected_information(no_has_word)
    return ent


def information_gain(word, sample):
    # used to calculate gain ratio
    sample_sizes = [len(sample[author]) for author in sample]
    return expected_information(sample_sizes) - entropy(word, sample)


def split_information(word, sample):
    # used to calculate gain ratio
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
        split += sum(has_word) / total_docs * math.log(sum(has_word) / total_docs, 2)
    if sum(no_has_word) != 0:
        split += sum(no_has_word) / total_docs * math.log(sum(no_has_word) / total_docs, 2)
    return -1 * split


def gain_ratio(word, sample):
    # used to determine which attribute to split on for C4.5
    split_info = split_information(word, sample)
    info_gain = information_gain(word, sample)
    if split_info == 0:
        return 0
    return info_gain / split_info


def c45(sample, depth, encountered_words, split_words):
    '''
    Make a single split in a decision tree, then calls itself recursively
    on the two sides of that split to create the full tree

    It will terminate once that tree has three layers, even though the tree will
    still be far from the complete decision tree
    '''
    best_ratio = (0.0, '')
    for word in encountered_words:
        if word in split_words:
            continue
        word_ratio = gain_ratio(word, sample)
        if word_ratio > best_ratio[0]:
            best_ratio = (word_ratio, word)
    if best_ratio[0] == 0:
        return []
    attr = best_ratio[1]
    # if we're adding another layer, we need to determine which documents
    # have the split attribute and which do not
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
            # recursive call on documents that have the attribute/word (which go to the left)
            left_words = c45(has_attr, depth + 1, encountered_words, new_split_words)
        if no_has_attr_length > 0:
            # recursive call on documents without the attribute/word
            right_words = c45(no_has_attr, depth + 1, encountered_words, new_split_words)
        # Return a list of this attribute plus all attributes split on in the lower
        # levels of the tree (since we don't actually care about the tree itself when we use these)
        return [attr] + left_words + right_words
    else:
        return [attr]


def main():

    # Run cross_validation on all documents with all features
    if sys.argv[1] == 'baseline':
        data_holder = naive_bayes_data.naive_bayes_data(writer_list)
        print 'Finished reading data'
        print cross_validation(5, data_holder.encountered_words, data_holder)
        return

    # Run our trials in groups of 3
    for i in range(3):
        data_holder = naive_bayes_data.naive_bayes_data(writer_list)
        writers, dev_data = split_10_data(data_holder.writers)
        data_holder.set_writers(writers)
        print("Finished reading data")

        # Use command line arguments to determine method to select features, then perform cross-validation
        features = set()
        if sys.argv[1] == 'naive':
            features = naive_feature_select(cutoffs[i], data_holder, dev_data)
        elif sys.argv[1] == 'greedy':
            max_iters = 20
            if len(sys.argv) > 2:
                max_iters = int(sys.argv[2])
            features = greedy_feature_select(data_holder, dev_data, max_iters)
        elif sys.argv[1] == 'random':
            num_features = 20
            if len(sys.argv) > 2:
                num_features = int(sys.argv[2])
            features = random.sample(data_holder.encountered_words,num_features)
        else: # decision tree
            data_holder = naive_bayes_data.naive_bayes_data(writer_list)  # we want to reset this to begin
            for j in range(10):
                writers, dev_data = split_10_data(data_holder.writers)
                features.update(set(c45(dev_data, 0, data_holder.encountered_words, [])))
            print '-------------------------------------------------------------'
        print ''
        print 'Selected features'
        print features
        print ''
        print ''
        print 'Selected features result'
        print cross_validation(5, features, data_holder)
        print ''
        print ''



if __name__ == '__main__':
    main()
