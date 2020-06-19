# Machine_Learning
COMP472 Artificial Intelligence Course Assignment 2. Classify posts by post type depending on the words in the title.

This assignment was coded on PyCharm using python 3.7
The easiest way to set up the assignment is of course by pulling it from github https://github.com/MahChaa/Machine_Learning

Once you opened the assignment folder or have pulled it from github on PyCharm, you'll need to set your environment.
If you don't have Python 3.7 installed on your PC, PyCharm can install a virtual environment in the work folder.
However, you'll have to figure it out on your own, it's a process that has nothing to do with my assignment.

In addition to that, you should get a notification on PyCharm, to install the requirements of the project.
If you don't get that notification, then you can open the requirements.txt files where all the requirements are listed and
install them manually.

To be able to run word_tokenize() from nltk, you'll have to download some data from nltk.data
It's simple, all you have to do is write this into the interpreter:
> > > import nltk
> > > nltk.download("punkt")
This will download the data at AppData\Roaming\nltk_data on your local machine (If you want to remove it after the fact)
If you're on Mac, it should still display the download path in the interpreter.

If all is good you should be able to run the main function without any issue, it's not rocket science.
