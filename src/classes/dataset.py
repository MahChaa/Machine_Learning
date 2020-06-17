# -------------------------------------------------------
# Assignment 2
# Written by Mahdi Chaari - 27219946
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
import math
import sys
from collections import Counter
import pandas as pd
import numpy as np
# word_tokenize will require
# >>> import nltk
# >>> nltk.download("punkt")
# to be input into the interpreter before it can work properly
from nltk import word_tokenize


class DataSet:
    model: dict
    training_words_frequency: Counter
    story_training_words: list
    story_training_words_frequency: dict
    ask_hn_training_words: list
    ask_hn_training_words_frequency: dict
    show_hn_training_words: list
    show_hn_training_words_frequency: dict
    poll_training_words: list
    poll_training_words_frequency: dict
    confusion_matrix: pd.DataFrame
    classification_accuracy: float

    def __init__(self, data_source: str, testing_set: str):
        # We skip the first column when creating the DataFrame since Pandas generates its own indices for all elements
        data = pd.read_csv("../resources/" + data_source, usecols=range(1, 10))
        data["Title"] = data["Title"].str.lower()  # All titles are made lowercase

        # We find the index of the first occurrence of the year 2019 (testing_Set)
        # in the Created At column assuming the data is sorted
        self.__test_sample_start_index = data["Created At"].values.searchsorted(testing_set, side='right')

        # In the case that there are no posts of 2019 (testing_Set), we abort the program
        if not self.__test_sample_start_index:
            sys.exit("There was an error splitting the data by year. No posts made in " + testing_set + " were found.")

        self.training_data = data.iloc[:self.__test_sample_start_index]
        self.testing_data = data.iloc[self.__test_sample_start_index:]

        # These are all the titles concatenated and separated with a space made into a list of all the words
        self.training_words = word_tokenize((self.training_data["Title"] + " ").sum())
        self.training_words.sort()

        # These are the initial frequencies used to smooth the probabilities, the frequencies will be added onto them
        self.training_types_frequency = {"story": 0.5, "ask_hn": 0.5, "show_hn": 0.5, "poll": 0.5}

        # Pandas Series have built in functions to get the frequencies of equal values
        # I turn it into a dictionary and iterate through it to extract the frequency of all post types
        for post_type, frequency in dict(self.training_data["Post Type"].value_counts()).items():
            self.training_types_frequency[post_type] += frequency

        self.training_data_size = int(self.training_data["Object ID"].count())

        self.training_types_probability = {"story": self.training_types_frequency["story"] / self.training_data_size,
                                           "ask_hn": self.training_types_frequency["ask_hn"] / self.training_data_size,
                                           "show_hn": self.training_types_frequency[
                                                          "show_hn"] / self.training_data_size,
                                           "poll": self.training_types_frequency["poll"] / self.training_data_size}

        # Using Pandas DataFrame built-in functions, all the words in titles of each post type are extracted separately
        self.story_training_words = (self.training_data[self.training_data["Post Type"] == "story"]["Title"] + " ") \
            .sum()

        # Splitting only occurs if the resulting variable is a string, if it is not a string then it is 0
        # which means that there was no occurrence found for that post type. We then assign to it an empty list
        self.story_training_words = word_tokenize(self.story_training_words) \
            if isinstance(self.story_training_words, str) else []

        # We create a Counter to count each word's frequency in titles separated by post types
        self.story_training_words_frequency = Counter(self.story_training_words)

        self.ask_hn_training_words = (self.training_data[self.training_data["Post Type"] == "ask_hn"]["Title"] + " ") \
            .sum()
        self.ask_hn_training_words = word_tokenize(self.ask_hn_training_words) \
            if isinstance(self.ask_hn_training_words, str) else []
        self.ask_hn_training_words_frequency = Counter(self.ask_hn_training_words)

        self.show_hn_training_words = (self.training_data[self.training_data["Post Type"] == "show_hn"]["Title"] + " ") \
            .sum()
        self.show_hn_training_words = word_tokenize(self.show_hn_training_words) \
            if isinstance(self.show_hn_training_words, str) else []
        self.show_hn_training_words_frequency = Counter(self.show_hn_training_words)

        self.poll_training_words = (self.training_data[self.training_data["Post Type"] == "poll"]["Title"] + " ").sum()
        self.poll_training_words = word_tokenize(self.poll_training_words) if isinstance(self.poll_training_words, str) \
            else []
        self.poll_training_words_frequency = Counter(self.poll_training_words)

        self.experiment_baseline()

    def create_model(self, file_name: str) -> None:
        # The model file is created here in a .txt file
        with open("../resources/" + file_name, "w", encoding="UTF-8") as file:
            # This variable is going to store all the data stored in the model file
            self.model = {}

            # We scan through all the vocabulary acquired from the dataset titles
            for index, (word, frequency) in enumerate(self.training_words_frequency.items()):
                # If the current word exists in the set of words posted under this specific type
                story_frequency = self.story_training_words_frequency.get(word, 0)

                # This is the formula of conditional probability as is in the lecture notes
                story_probability = (story_frequency + 0.5) / (len(self.story_training_words_frequency)
                                                               + (0.5 * len(self.training_words_frequency)))

                ask_hn_frequency = self.ask_hn_training_words_frequency.get(word, 0)
                ask_hn_probability = (ask_hn_frequency + 0.5) / (len(self.ask_hn_training_words_frequency)
                                                                 + (0.5 * len(self.training_words_frequency)))

                show_hn_frequency = self.show_hn_training_words_frequency.get(word, 0)
                show_hn_probability = (show_hn_frequency + 0.5) / (len(self.show_hn_training_words_frequency)
                                                                   + (0.5 * len(self.training_words_frequency)))

                poll_frequency = self.poll_training_words_frequency.get(word, 0)
                poll_probability = (poll_frequency + 0.5) / (len(self.poll_training_words_frequency)
                                                             + (0.5 * len(self.training_words_frequency)))

                self.model.update({word: (story_frequency, story_probability, ask_hn_frequency, ask_hn_probability,
                                          show_hn_frequency, show_hn_probability, poll_frequency, poll_probability)})

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

    def classify(self, file_name: str) -> None:
        self.confusion_matrix = pd.DataFrame(
            {"story": [0, 0, 0, 0], "ask_hn": [0, 0, 0, 0], "show_hn": [0, 0, 0, 0], "poll": [0, 0, 0, 0]},
            index=["story", "ask_hn", "show_hn", "poll"])

        # The classification results file is created here in a .txt file
        with open("../resources/" + file_name, "w", encoding="UTF-8") as file:

            # The testing DataFrame is iterated through to extract all post titles
            for index, row in self.testing_data.iterrows():
                score = {"story": 0, "ask_hn": 0, "show_hn": 0, "poll": 0}

                for word in word_tokenize(row["Title"]):
                    if self.model.__contains__(word):
                        score["story"] += math.log(self.model[word][1])
                        score["ask_hn"] += math.log(self.model[word][3])
                        score["show_hn"] += math.log(self.model[word][5])
                        score["poll"] += math.log(self.model[word][7])

                score["story"] += math.log(self.training_types_probability["story"])
                score["ask_hn"] += math.log(self.training_types_probability["ask_hn"])
                score["show_hn"] += math.log(self.training_types_probability["show_hn"])
                score["poll"] += math.log(self.training_types_probability["poll"])

                post_type_prediction = max(score["story"], score["ask_hn"], score["show_hn"], score["poll"])

                if post_type_prediction == score["story"]:
                    post_type_prediction = "story"
                elif post_type_prediction == score["ask_hn"]:
                    post_type_prediction = "ask_hn"
                elif post_type_prediction == score["show_hn"]:
                    post_type_prediction = "show_hn"
                else:
                    post_type_prediction = "poll"

                self.confusion_matrix.loc[row["Post Type"], post_type_prediction] += 1

                print(index - self.__test_sample_start_index + 1, row["Title"], post_type_prediction,
                      "%.7f" % score["story"],
                      "%.7f" % score["ask_hn"],
                      "%.7f" % score["show_hn"],
                      "%.7f" % score["poll"],
                      row["Post Type"],
                      "right" if post_type_prediction == row["Post Type"] else "wrong",
                      sep="  ", file=file)

            self.classification_accuracy = (np.diag(self.confusion_matrix).sum() / self.training_data_size) * 100

    def experiment_baseline(self) -> None:
        self.training_words_frequency = Counter(self.training_words)

    def experiment_1(self, file_name: str) -> None:
        self.experiment_baseline()

        stop_words = list(pd.read_table("../resources/" + file_name, names=["Stop Words"])["Stop Words"])

        for word in stop_words:
            if word in self.training_words_frequency:
                del self.training_words_frequency[word]

    def experiment_2(self, min_size: int, max_size: int):
        self.experiment_baseline()

        for word in list(self.training_words_frequency.keys()):
            if not min_size <= len(word) <= max_size:
                del self.training_words_frequency[word]

    def experiment_3(self):
        self.experiment_baseline()


