TODO:
- tf-idf
- run:
    - naive w/ diff cut-off (0.32, 0.4, 0.5) x 2
        - run ending good features with cross validation for each
    - greedy x 6
        - plot one run of num words vs result - have
        - run ending good features with cross validation for each
    - decision x 6
        - run ending good features with cross validation for each



Decisions:
- only run greedy for max 20 words, because otherwise takes too long
- start naive cut-off at 0.32 because too many words as 0.319