# -------------------------------------------------------
# Assignment 2
# Written by Mahdi Chaari - 27219946
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
import pandas as pd


def main():
    print("Hello and welcome!")

    # We skip the first column when creating the DataFrame since Pandas generates its own indices for all elements
    data = pd.read_csv("../resources/hns_2018_2019.csv", usecols=range(1, 10))

    # We find the index of the first occurrence of the year 2019 in the Created At column assuming the data is sorted
    test_sample_start_index = data["Created At"].values.searchsorted('2019', side='right')

    print(test_sample_start_index)


if __name__ == "__main__":
    main()
