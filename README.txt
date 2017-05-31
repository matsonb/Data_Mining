main.py, naive_bayes_data.py, and the four data files (austen-parsed.txt, dickens-parsed.txt, shakespeare-parsed.txt, and et-al-parsed.txt) should all be in the same directory. To run the Naive Bayes classifier on the data, the general form of the command line input is "python main.py METHOD" where METHOD is one of "baseline," "naive," "greedy," "random," or "tree" (although "tree" is never checked for since it is caught in an else block, so any string besides the previous four could be used to have the same effect). These have the following effects:

baseline: run 5-fold cross-validation Naive Bayes on the entire data set with all of the possible attributes, to get a baseline to compare the other methods to
naive: run "naive" Naive Bayes, where the cutoffs used to determine the selected features come from an array specified on line 7 of main.py. (This will use the first 3 entries of that list, so it must have length at least 3 but all entries beyond the 3rd will be ignored)
greedy: run the greedy subset selection algorithm. This has an optional parameter of an integer to determine the maximum number of iterations (or features to be selected), which defaults to 20 if absent.
random: randomly select some features to use as an alternative baseline. As with greedy, this defaults to 20 unless an optional parameter of an integer is included at the end of the command line arguments to override that.
tree: run the decision tree selection algorithm to select the features used in Naive Bayes classification

In all of the cases except for baseline, this will run the specified algorithm three times and then run 5-fold cross-validation on the (non-development part of the) data to determine the accuracy of the classifier.

The largest known issue is that the code can take a long time to run (especially for greedy). Running the code with pypy empirically has helped with a lot with that, but still running greedy with a large number for max_iters can take a long time.