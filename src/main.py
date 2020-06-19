# -------------------------------------------------------
# Assignment 2
# Written by Mahdi Chaari - 27219946
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
from classes.dataset import DataSet


def main():
    print("Hello and welcome!")

    data = DataSet("hns_2018_2019.csv", "2019")

    data.create_model("model-2018.txt")


if __name__ == "__main__":
    main()
