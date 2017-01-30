from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from math import log

class NGramModel(object):
    """
    Takes a series from a DataFrame and an n and trains a model of a given n.

    Parameters
    ----------
    n : int
        Size of the window to look at my data.

    Attributes
    ----------
    n_grams_count : defaultdict of Counter()
        Counts the occurance of the last character in a window to the preceding ones.
    frequencies : defaultdict of defaultdicts.
        Holds the frequency a given last character shows up after the preceding ones given that all chararters show up.
    perplexity : float
        Holds the overall perplexity of my model given a new piece of data.

    """
    def __init__(self, n):
        self.n = n
        # dictionary of counters that take a prefix and count all occuranced of what comes after.
        self.n_grams_count = None
        # defaultdict of defaultdict that holds the frequencies of a given suffix.
        self.frequencies = None

    def get_window_properties(tune,i):
        window = tune[i:i + self.n]
        given = window[:self.n-1]
        following = window[-1]
        return given, following

    def fit(self, series):
        # For one character at a time.
        if self.n == 1:
            self.n_grams_count = Counter()
            for tune in series.values:
                self.n_grams_count += Counter(tune)
            self._create_frequency()

        # for more than 1
        if self.n > 1:
            # Create dictionary of Counter() with the n-gram as key and the
            # follwing next letter as key and the count of that occuring as
            # value.
            self.n_grams_count = defaultdict(lambda: Counter()) #R.I.: why is the lambda necessary? I think it can be written without it
            # Go through each tune.
            for tune in series.values:
                tune = self._pad_string(tune)

                #for i, sound in enumerate(tune[:-self.n]): (maybe personal choice, but I think enumerate tends to look a bit cleaner than xrange)
                for i in xrange(len(tune) - self.n + 1):
                    given, following = self.get_window_properties(tune,i)
                    self.n_grams_count[given][following] += 1

            self._create_frequency()


    def _pad_string(self, string): # R.I.: add docstring, not sure why string needs to be padded
        if self.n == 1:
            return string
        else:
            #pad with n-1
            pad = '?' * (self.n - 1)
            string = pad + string + pad
            return string

    def get_frequency_dict(counter):
        total_n = float(sum(counter.values()))
        return {k: v/total_n for k, v in counter.iteritems()}

    # Calculates the frequency of a given n-gram.
    def _create_frequency(self):
        n_total = 0
        #for n size 1 its simpler and faster.
        if self.n == 1:
            self.frequencies = defaultdict(lambda: 0.0) # is this not the same as defaultdict(float) ?
            #predict method.
            frequencies = self.get_frequency_dict(self.n_grams_count)
            self.frequencies.update(frequencies)

        if self.n >1:
            # Create a dictionary of frequencies that calculates the
            # occurance of a letter after an n-gram and returns the
            # frequency of that. Update self.frequencies
            total_n = 0
            self.frequencies = defaultdict(lambda: defaultdict(lambda: 0.0)) #again dont think the lambdas are necessary
            for given, counter in self.n_grams_count.iteritems():
                frequencies = {}
                frequencies[given] = self.get_frequency_dict(counter)
                self.frequencies.update(frequencies)

#Doesn't work yet.
#Returns the perplexity of a given model.
    def perplexity_score(self, new_tune):
        #import pdb; pdb.set_trace()
        sum_of_log_probs = 0
        working_tune = self._pad_string(new_tune)
        for i in xrange(len(working_tune) - self.n): # for i, sound in enumerate(working_tune[:-self.n])
            history, token = self.get_window_properties(working_tune,i)
            probability = self.frequencies[history][token]
            sum_of_log_probs += log(probability + 1e-10)

        perplexity = 1 - sum_of_log_probs
        return perplexity
