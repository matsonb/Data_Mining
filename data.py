import string
import csv
import re

pattern = re.compile('([^\s\w]|_)+')


def make_csv(filename):
    file = open(filename + '.txt','r')
    delimiter = ' '
    text = delimiter.join([line.strip() for line in file])
    file.close()
    to_remove = string.letters + ' '
    new_to_remove = text.translate(None,to_remove)
    new_text = text.translate(None,new_to_remove)


    words = new_text.split(' ')

    words = [words[i].lower() for i in range(len(words)) if len(words[i]) > 0]

    shingles = [words[i:i+1000] for i in range(0,len(words),1000)]

    file = open(filename + '-parsed.txt','w')
    csv_file = csv.writer(file, delimiter=',')
    lines = 0
    for shingle in shingles:
        if lines == 1000:
            break
        lines += 1
        csv_file.writerow(shingle)
    file.close()

def main():
    make_csv('shakespeare')
    print('shakespeare')
    make_csv('austen')
    print('austen')
    make_csv('dickens')
    print('dickens')
    make_csv('et-al')


if __name__ == '__main__':
    main()
