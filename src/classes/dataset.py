# -------------------------------------------------------
# Assignment 2
# Written by Mahdi Chaari - 27219946
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
import pandas as pd
import sys
# word_tokenize will require
# >>> import nltk
# >>> nltk.download("punkt")
# to be input into the interpreter before it can work properly
from nltk import word_tokenize
from collections import Counter


class DataSet:
    def __init__(self, data_source: str, testing_set: str):
        # We skip the first column when creating the DataFrame since Pandas generates its own indices for all elements
        data = pd.read_csv("../resources/" + data_source, usecols=range(1, 10))
        data["Title"] = data["Title"].str.lower()  # All titles are made lowercase

        # We find the index of the first occurrence of the year 2019 (testing_Set)
        # in the Created At column assuming the data is sorted
        test_sample_start_index = data["Created At"].values.searchsorted(testing_set, side='right')

        # In the case that there are no posts of 2019 (testing_Set), we abort the program
        if not test_sample_start_index:
            sys.exit("There was an error splitting the data by year. No posts made in " + testing_set + " were found.")

        self.training_data = data.iloc[:test_sample_start_index]
        self.testing_data = data.iloc[test_sample_start_index:]

        # These are all the titles concatenated and separated with a space made into a list of all the words
        self.training_words = word_tokenize((self.training_data["Title"] + " ").sum())
        self.training_words.sort()

        self.testing_words = word_tokenize((self.testing_data["Title"] + " ").sum())
        self.testing_words.sort()

        # removed_words = []

        # for index, word in enumerate(words_2018):
        #     while not 97 <= ord(word[-1]) <= 122:
        #         word = word[:-1]
        #         words_2018[index] = word

        self.training_words_frequency = Counter(self.training_words)
        self.testing_words_frequency = Counter(self.testing_words)

        # These are the initial frequencies used to smooth the probabilities, the frequencies will be added onto them
        self.training_types_frequency = {"story": 0.5, "ask_hn": 0.5, "show_hn": 0.5, "poll": 0.5}
        self.testing_types_frequency = {"story": 0.5, "ask_hn": 0.5, "show_hn": 0.5, "poll": 0.5}

        # Pandas Series have built in functions to get the frequencies of equal values
        # I turn it into a dictionary and iterate through it to extract the frequency of all post types
        for post_type, frequency in dict(self.training_data["Post Type"].value_counts()).items():
            self.training_types_frequency[post_type] += frequency

        for post_type, frequency in dict(self.testing_data["Post Type"].value_counts()).items():
            self.testing_types_frequency[post_type] += frequency

        self.training_data_size = int(self.training_data["Object ID"].count())
        self.testing_data_size = int(self.testing_data["Object ID"].count())

        # Using Pandas DataFrame built-in functions, all the words in titles of each post type are extracted separately
        self.story_training_words = (self.training_data[self.training_data["Post Type"] == "story"]["Title"] + " ") \
            .sum()

        # Splitting only occurs if the resulting variable is a string, if it is not a string then it is 0
        # which means that there was no occurrence found for that post type. We then assign to it an empty list
        self.story_training_words = word_tokenize(self.story_training_words)\
            if isinstance(self.story_training_words, str) else []

        # We create a Counter to count each word's frequency in titles separated by post types
        self.story_training_words_frequency = Counter(self.story_training_words)

        self.ask_hn_training_words = (self.training_data[self.training_data["Post Type"] == "ask_hn"]["Title"] + " ") \
            .sum()
        self.ask_hn_training_words = word_tokenize(self.ask_hn_training_words)\
            if isinstance(self.ask_hn_training_words, str) else []
        self.ask_hn_training_words_frequency = Counter(self.ask_hn_training_words)

        self.show_hn_training_words = (self.training_data[self.training_data["Post Type"] == "show_hn"]["Title"] + " ") \
            .sum()
        self.show_hn_training_words = word_tokenize(self.show_hn_training_words) \
            if isinstance(self.show_hn_training_words, str) else []
        self.show_hn_training_words_frequency = Counter(self.show_hn_training_words)

        self.poll_training_words = (self.training_data[self.training_data["Post Type"] == "poll"]["Title"] + " ").sum()
        self.poll_training_words = word_tokenize(self.poll_training_words) if isinstance(self.poll_training_words, str) else []
        self.poll_training_words_frequency = Counter(self.poll_training_words)

    def create_model(self, file_name: str) -> None:
        # The model file is created here in a .txt file
        with open("../resources/" + file_name, "w", encoding="UTF-8") as file:

            # We scan through all the vocabulary acquired from the dataset titles
            for index, (word, frequency) in enumerate(self.training_words_frequency.items()):

                # If the current word exists in the set of words posted under this specific type
                story_frequency = self.story_training_words_frequency[word] \
                    if self.story_training_words_frequency.__contains__(word) else 0

                # This is the formula of conditional probability as is in the lecture notes
                story_probability = (story_frequency + 0.5) / (len(self.story_training_words_frequency)
                                                               + (0.5 * len(self.training_words_frequency)))

                ask_hn_frequency = self.ask_hn_training_words_frequency[word] \
                    if self.ask_hn_training_words_frequency.__contains__(word) else 0
                ask_hn_probability = (ask_hn_frequency + 0.5) / (len(self.ask_hn_training_words_frequency)
                                                                 + (0.5 * len(self.training_words_frequency)))

                show_hn_frequency = self.show_hn_training_words_frequency[word] \
                    if self.show_hn_training_words_frequency.__contains__(word) else 0
                show_hn_probability = (show_hn_frequency + 0.5) / (len(self.show_hn_training_words_frequency)
                                                                   + (0.5 * len(self.training_words_frequency)))

                poll_frequency = self.poll_training_words_frequency[word] \
                    if self.poll_training_words_frequency.__contains__(word) else 0
                poll_probability = (poll_frequency + 0.5) / (len(self.poll_training_words_frequency)
                                                             + (0.5 * len(self.training_words_frequency)))

                print(index + 1, word,
                      story_frequency,
                      "%.7f" % story_probability,
                      ask_hn_frequency,
                      "%.7f" % ask_hn_probability,
                      show_hn_frequency,
                      "%.7f" % show_hn_probability,
                      poll_frequency,
                      "%.7f" % poll_probability,
                      sep="  ", file=file)
