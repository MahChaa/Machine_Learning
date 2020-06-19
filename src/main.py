# -------------------------------------------------------
# Assignment 2
# Written by Mahdi Chaari - 27219946
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
import time
from classes.dataset import DataSet
import matplotlib.pyplot as plt


def main():
    print("Hello and welcome!")

    start_time = time.time()

    data = DataSet("hns_2018_2019.csv", "2019")

    loading_time = time.time()

    data.create_model("model-2018.txt")
    data.classify("baseline-result.txt")

    print("\nBaseline confusion matrix (%.3f%% correct):\n" % data.classification_accuracy, data.confusion_matrix)

    print("\nTime to perform baseline experiment: %.3f seconds" % (time.time() - loading_time))

    loading_time = time.time()

    data.experiment_1("stopwords.txt")
    data.create_model("stopword-model.txt")
    data.classify("stopword-result.txt")

    print("\nExperiment 1 confusion matrix (%.3f%% correct):\n" % data.classification_accuracy, data.confusion_matrix)

    print("\nTime to perform experiment 1: %.3f seconds" % (time.time() - loading_time))

    loading_time = time.time()

    data.experiment_2(3, 8)
    data.create_model("wordlength-model.txt")
    data.classify("wordlength-result.txt")

    print("\nExperiment 2 confusion matrix (%.3f%% correct):\n" % data.classification_accuracy, data.confusion_matrix)

    print("\nTime to perform experiment 2: %.3f seconds" % (time.time() - loading_time))

    loading_time = time.time()

    data.experiment_3([1, 5, 10, 15, 20], [5, 10, 15, 20, 25])

    print("\nTime to perform experiment 3: %.3f seconds" % (time.time() - loading_time))

    plt.show()

    print("\nThank you. Goodbye.")


if __name__ == "__main__":
    main()
