import csv

encountered_words = set()

file = open('austen-parsed.txt')
csv_file = csv.reader(file)

for row in csv_file:
    for word in row:
        encountered_words.add(word)
file.close()       
print(1)

file = open('shakespeare-parsed.txt')
csv_file = csv.reader(file)

for row in csv_file:
    for word in row:
        encountered_words.add(word)
file.close()
print(2)

file = open('dickens-parsed.txt')
csv_file = csv.reader(file)
for row in csv_file:
    for word in row:
        encountered_words.add(word)
file.close()

print(3)
file = open('et-al-parsed.txt')
csv_file = csv.reader(file)
for row in csv_file:
    for word in row:
        encountered_words.add(word)
file.close()
        
print(len(encountered_words))
