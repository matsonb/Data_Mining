import math
import random
import naive_bayes_data


writer_list = ['austen', 'dickens', 'shakespeare', 'et-al']


def split_10_data(full_data):
    """
    Randomly removes 10% of the data for purposes of testing parameters and features
    """
    data_10_per = {}
    data_90_per = {}
    for writer in writer_list:
        doc_list = list(full_data[writer])
        docs_10_per = [doc_list.pop(random.randrange(len(doc_list))) for i in range(int(len(doc_list)/10.0))]
        data_10_per[writer] = docs_10_per
        data_90_per[writer] = doc_list
    return data_90_per, data_10_per


def naive_feature_select(cutoff, data_holder, dev_data):
    """
    Finds a smaller subset of features to use by
    finding every single feature that performs the best.
    TODO: run this a bunch of times
    """
    dev_data_size = sum([len(values) for values in dev_data.itervalues])
    good_features = set()
    for word in data_holder.encountered_words:
        correct = 0.0
        for writer in writer_list:
            for doc in dev_data[writer]:
                if data_holder.naive_bayes(doc, {word}) == writer:
                    correct += 1
        if correct / dev_data_size > cutoff:
            good_features.add(word)
    
    final_correct = 0.0
    for writer in dev_data:
        for doc in dev_data[writer]:
            if data_holder.naive_bayes(doc, good_features) == writer:
                final_correct += 1

    print(good_features)
    print(len(good_features))
    print(final_correct / dev_data_size)


def greedy_feature_select(data_holder, dev_data):
    """
    Finds a smaller subset of features to use by
    iteratively choosing next feature to add on by choosing the one that improves
    the current set the most

    Stops when the the addition of any remaining feature would decrease the accuracy
    """
    s = [0.0, set()]
    unused_words = data_holder.encountered_words
    while len(s[1]) < 20:
        print("")
        print("")
        print("")
        best_t = [0,set(), '']
        for word in unused_words:
            t = s[1].union({word})
            t_score = 0.0
            for writer in writer_list:
                for doc in dev_data[writer]:
                    if data_holder.naive_bayes(doc, t) == writer:
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


def all_features(data_holder, dev_data):
    correct = 0.0
    total = 0.0
    for writer in writer_list:
        for doc in dev_data[writer]:
            if data_holder.naive_bayes(doc) == writer:
                correct += 1
            total += 1
    return correct


def create_batches(num_batches, writers):
    """ defines random mini-batches of the training data """
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


def cross_validation(num_batches, features, data_holder):
    """ runs cross validation naive bayes on the data with the given features and number of batches """
    features = set(features)
    batches = create_batches(num_batches, data_holder.writers)
    print("Finished creating batches")
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
        print("Finished test " + str(test+1))
    return correct/total


def main():
    data_holder = naive_bayes_data.naive_bayes_data(writer_list)
    writers, dev_data = split_10_data(data_holder.writers)
    data_holder.set_writers(writers)
    print("Finished reading data")
    print all_features(data_holder, dev_data)
    feature_words = ['hath', 'example', 'feelings', 'manners', 'thy', 'allow', 'b', 'dorrit', 'friendship',
                     'carriage', 'cant', 'plan', 'handed', 'conduct', 'candle', 'her', 'crows', 'repeated',
                     'was', 'faded', 'thin', 'excite']
    print cross_validation(5, feature_words, data_holder)
    print cross_validation(5, data_holder.encountered_words, data_holder)

if __name__ == '__main__':
    main()

