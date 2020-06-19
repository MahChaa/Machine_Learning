# -------------------------------------------------------
# Assignment 2
# Written by Mahdi Chaari - 27219946
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
import math
import sys
from collections import Counter
import io
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

        self.ask_hn_training_words = (self.training_data[self.training_data["Post Type"] == "ask_hn"]["Title"] + " ") \
            .sum()
        self.ask_hn_training_words = word_tokenize(self.ask_hn_training_words) \
            if isinstance(self.ask_hn_training_words, str) else []

        self.show_hn_training_words = (self.training_data[self.training_data["Post Type"] == "show_hn"]["Title"] + " ") \
            .sum()
        self.show_hn_training_words = word_tokenize(self.show_hn_training_words) \
            if isinstance(self.show_hn_training_words, str) else []

        self.poll_training_words = (self.training_data[self.training_data["Post Type"] == "poll"]["Title"] + " ").sum()
        self.poll_training_words = word_tokenize(self.poll_training_words) if isinstance(self.poll_training_words, str) \
            else []

        self.experiment_baseline()

        # Setting for showing windowed graph using matplotlib
        matplotlib.use("TkAgg")

    def create_model(self, file_name: str = None) -> None:
        # This variable is going to store all the data gathered for the model file
        self.model = {}

        # The model data is created here before being flushed in a .txt file
        if file_name is not None:
            output = io.StringIO()

        # We scan through all the vocabulary acquired from the dataset titles
        for index, (word, frequency) in enumerate(self.training_words_frequency.items()):
            # If the current word exists in the set of words posted under this specific type
            story_frequency = self.story_training_words_frequency.get(word, 0)

            # This is the formula of conditional probability as is in the lecture notes
            story_probability = (story_frequency + 0.5) / (sum(self.story_training_words_frequency.values())
                                                           + (0.5 * len(self.training_words_frequency)))

            ask_hn_frequency = self.ask_hn_training_words_frequency.get(word, 0)
            ask_hn_probability = (ask_hn_frequency + 0.5) / (sum(self.ask_hn_training_words_frequency.values())
                                                             + (0.5 * len(self.training_words_frequency)))

            show_hn_frequency = self.show_hn_training_words_frequency.get(word, 0)
            show_hn_probability = (show_hn_frequency + 0.5) / (sum(self.show_hn_training_words_frequency.values())
                                                               + (0.5 * len(self.training_words_frequency)))

            poll_frequency = self.poll_training_words_frequency.get(word, 0)
            poll_probability = (poll_frequency + 0.5) / (sum(self.poll_training_words_frequency.values())
                                                         + (0.5 * len(self.training_words_frequency)))

            self.model.update({word: (story_frequency, story_probability, ask_hn_frequency, ask_hn_probability,
                                      show_hn_frequency, show_hn_probability, poll_frequency, poll_probability)})

            # If we have a file name, the data is printed line by line at every iteration for each word
            if file_name is not None:
                print(index + 1, word,
                      story_frequency,
                      "%.7f" % story_probability,
                      ask_hn_frequency,
                      "%.7f" % ask_hn_probability,
                      show_hn_frequency,
                      "%.7f" % show_hn_probability,
                      poll_frequency,
                      "%.7f" % poll_probability,
                      sep="  ", file=output)

        # If we have a file name, the string stream filled with data from the previous loop is flushed in to the file
        if file_name is not None:
            with open("../models/" + file_name, "w", encoding="UTF-8") as file:
                print(output.getvalue(), file=file, end="")

    def classify(self, file_name: str = None) -> None:
        # This will be used to visualize the accuracy of the predictions
        self.confusion_matrix = pd.DataFrame(
            {"story": [0, 0, 0, 0], "ask_hn": [0, 0, 0, 0], "show_hn": [0, 0, 0, 0], "poll": [0, 0, 0, 0]},
            index=["story", "ask_hn", "show_hn", "poll"])

        # The classification results file is created here before being flushed in a .txt file
        if file_name is not None:
            output = io.StringIO()

        # The testing DataFrame is iterated through to extract all post titles
        for index, row in self.testing_data.iterrows():
            score = {"story": 0, "ask_hn": 0, "show_hn": 0, "poll": 0}

            # The title is broken down into tokens
            for word in word_tokenize(row["Title"]):
                # The probability for the word to be of each post type is read from the model variable and summed
                if self.model.__contains__(word):
                    score["story"] += math.log(self.model[word][1])
                    score["ask_hn"] += math.log(self.model[word][3])
                    score["show_hn"] += math.log(self.model[word][5])
                    score["poll"] += math.log(self.model[word][7])

            # The probability of the post type itself is added at the end
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

            # This populated the confusion matrix adding one where it is relevant (y = true class, x = predicted class)
            self.confusion_matrix.loc[row["Post Type"], post_type_prediction] += 1

            # Same thing as for the model creation in terms of the string stream
            if file_name is not None:
                print(index - self.__test_sample_start_index + 1, row["Title"], post_type_prediction,
                      "%.7f" % score["story"],
                      "%.7f" % score["ask_hn"],
                      "%.7f" % score["show_hn"],
                      "%.7f" % score["poll"],
                      row["Post Type"],
                      "right" if post_type_prediction == row["Post Type"] else "wrong",
                      sep="  ", file=output)

        if file_name is not None:
            with open("../results/" + file_name, "w", encoding="UTF-8") as file:
                print(output.getvalue(), file=file, end="")

        # This is the accuracy in percentage
        self.classification_accuracy = (np.diag(self.confusion_matrix).sum() / self.training_data_size) * 100

    def experiment_baseline(self) -> None:
        removed_words = []

        self.training_words_frequency = Counter(self.training_words)

        # We create a Counter to count each word's frequency in titles separated by post types
        self.story_training_words_frequency = Counter(self.story_training_words)
        self.ask_hn_training_words_frequency = Counter(self.ask_hn_training_words)
        self.show_hn_training_words_frequency = Counter(self.show_hn_training_words)
        self.poll_training_words_frequency = Counter(self.poll_training_words)

        # All words that aren't fully alphabetical are removed
        for word in list(self.training_words_frequency.keys()):
            if not word.isalpha():
                del self.training_words_frequency[word]

                # They are also removed from each post type's dictionary
                if self.story_training_words_frequency.__contains__(word):
                    del self.story_training_words_frequency[word]
                if self.ask_hn_training_words_frequency.__contains__(word):
                    del self.ask_hn_training_words_frequency[word]
                if self.show_hn_training_words_frequency.__contains__(word):
                    del self.show_hn_training_words_frequency[word]
                if self.poll_training_words_frequency.__contains__(word):
                    del self.poll_training_words_frequency[word]

                removed_words.append(word)

        # The removed_words list is printed to  a file if it contains anything
        if len(removed_words) > 0:
            with open("../resources/remove_word.txt", "w", encoding="UTF-8") as file:
                for word in removed_words:
                    print(word, file=file)

    def experiment_1(self, file_name: str) -> None:
        # Making sure we start with the baseline
        self.experiment_baseline()

        # A Series is created out stopwords.txt
        stop_words = pd.read_table("../resources/" + file_name, names=["Stop Words"])["Stop Words"]

        # All words in the Series are removed from the vocabulary
        for word in stop_words:
            if word in self.training_words_frequency:
                del self.training_words_frequency[word]

                if self.story_training_words_frequency.__contains__(word):
                    del self.story_training_words_frequency[word]
                if self.ask_hn_training_words_frequency.__contains__(word):
                    del self.ask_hn_training_words_frequency[word]
                if self.show_hn_training_words_frequency.__contains__(word):
                    del self.show_hn_training_words_frequency[word]
                if self.poll_training_words_frequency.__contains__(word):
                    del self.poll_training_words_frequency[word]

    def experiment_2(self, min_size: int, max_size: int):
        self.experiment_baseline()

        # All words smaller or bigger than the arguments are removed from the vocabulary
        for word in list(self.training_words_frequency.keys()):
            if not min_size <= len(word) <= max_size:
                del self.training_words_frequency[word]

                if self.story_training_words_frequency.__contains__(word):
                    del self.story_training_words_frequency[word]
                if self.ask_hn_training_words_frequency.__contains__(word):
                    del self.ask_hn_training_words_frequency[word]
                if self.show_hn_training_words_frequency.__contains__(word):
                    del self.show_hn_training_words_frequency[word]
                if self.poll_training_words_frequency.__contains__(word):
                    del self.poll_training_words_frequency[word]

    def experiment_3(self, frequency_threshold: list, percentile_threshold: list):
        self.experiment_baseline()

        frequency_threshold.sort()  # Lists are sorted just in case
        percentile_threshold.sort()

        performance = []
        vocabulary_size = []

        # The experiment is repeated with a trimmed vocabulary at each iteration including the baseline at first
        for threshold in frequency_threshold:
            for word, frequency in dict(self.training_words_frequency).items():
                if frequency <= threshold:
                    del self.training_words_frequency[word]

                    if self.story_training_words_frequency.__contains__(word):
                        del self.story_training_words_frequency[word]
                    if self.ask_hn_training_words_frequency.__contains__(word):
                        del self.ask_hn_training_words_frequency[word]
                    if self.show_hn_training_words_frequency.__contains__(word):
                        del self.show_hn_training_words_frequency[word]
                    if self.poll_training_words_frequency.__contains__(word):
                        del self.poll_training_words_frequency[word]

            self.create_model()
            self.classify()

            # The performance and the vocabulary size are saved in lists to be used in plotting as x and y
            performance.append(self.classification_accuracy)
            vocabulary_size.append(len(self.training_words_frequency))

        fig, plot = plt.subplots()

        fig.suptitle("Words Removed on the Basis of Specific Frequencies")

        plot.plot(vocabulary_size, performance, "-bo")

        plt.gca().invert_xaxis()

        plot.set_xlabel("Vocabulary Size")
        plot.set_ylabel("Classification Accuracy (%)")

        plt.show(block=False)

        # The lists are cleared to be reused
        performance.clear()
        vocabulary_size.clear()

        # The vocabulary is set back to the baseline before the second part of the experiment starts
        self.experiment_baseline()

        # Since elements will be deleted from the vocabulary, its size should be saved as a reference
        training_vocab_size = len(self.training_words_frequency)

        for threshold in percentile_threshold:
            # The percentage is calculated from the list size as an index (all words below that index are deleted)
            # The index is always calculated relatively to the original size of the vocabulary
            threshold_index = math.ceil(len(self.training_words_frequency)
                                        - (training_vocab_size - (training_vocab_size * threshold / 100)))

            # Here the list of tuples (word, frequency) is sorted from most common to least so the threshold percentage
            # can be removed from the beginning of the list using  threshold_index
            for element in self.training_words_frequency.most_common()[:threshold_index]:
                del self.training_words_frequency[element[0]]

                if self.story_training_words_frequency.__contains__(element[0]):
                    del self.story_training_words_frequency[element[0]]
                if self.ask_hn_training_words_frequency.__contains__(element[0]):
                    del self.ask_hn_training_words_frequency[element[0]]
                if self.show_hn_training_words_frequency.__contains__(element[0]):
                    del self.show_hn_training_words_frequency[element[0]]
                if self.poll_training_words_frequency.__contains__(element[0]):
                    del self.poll_training_words_frequency[element[0]]

            self.create_model()
            self.classify()
            performance.append(self.classification_accuracy)
            vocabulary_size.append(len(self.training_words_frequency))

        fig, plot = plt.subplots()

        fig.suptitle("Words Removed on the Basis of Top Percentage")

        plot.plot(vocabulary_size, performance, "-bo")

        plt.gca().invert_xaxis()

        plot.set_xlabel("Vocabulary Size")
        plot.set_ylabel("Classification Accuracy (%)")

        plt.show(block=False)
