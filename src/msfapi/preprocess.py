import numpy as np
import pandas as pd

def preprocess():
    df = pd.read_csv('2017-regularLabeledRaw.csv')
    print(df)


if __name__ == '__main__':
    preprocess()