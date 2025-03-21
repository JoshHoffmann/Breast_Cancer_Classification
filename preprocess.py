import plotting

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def ReadData()->pd.DataFrame:
    '''Read raw data from csv file and drop unnecessary id column'''
    data = pd.read_csv('data.csv')
    data = data.drop(columns=['id'])
    return data

def EncodeDiagnosis(data:pd.DataFrame)->pd.DataFrame:
    '''Encode the categorical diagnosis data'''
    encoder = LabelEncoder()
    encoder.fit(['M', 'B'])
    data.diagnosis = encoder.transform(data.diagnosis)

    return data

def Outliers(data:pd.DataFrame):
    features = data.drop(columns='diagnosis')
    def GetOutlierMask():
        median = features.median()
        MAD = (features-median).abs().median()
        adjusted_z = (features - median)/(1.4862*MAD)
        mask = (adjusted_z.abs() <= 4).all(axis=1)

        return mask
    mask = GetOutlierMask()
    outlier_cleaned_data = data[mask]
    return outlier_cleaned_data


def GetData():
    '''Get preprocessed data set'''
    data = ReadData()
    data = EncodeDiagnosis(data)
    data = Outliers(data)

    return data

