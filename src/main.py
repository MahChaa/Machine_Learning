# -------------------------------------------------------
# Assignment 2
# Written by Mahdi Chaari - 27219946
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
import pandas as pd


def main():
    print("Hello and welcome!")

    # We skip the first column when creating the DataFrame since Pandas generates its own indexes for all elements
    data = pd.read_csv("../resources/hns_2018_2019.csv", usecols=range(1, 10))

    # We iterate through the DataFrame until we come across the first 2019 entry, we save its index for future purposes
    for index, row in data.iterrows():
        if row["Created At"][:4] == "2019":
            test_data_index_start = index
            break


if __name__ == "__main__":
    main()
