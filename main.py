import pandas as pd
import numpy as np


if __name__ == '__main__':

    # read data
    data = pd.read_csv("Employee.csv")

    # split into training and testing dataset
    training_data, testing_data = train_test_split(data, )


