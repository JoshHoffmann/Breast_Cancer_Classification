import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import plotting
import preprocess
import modelling

def main():

    data = preprocess.GetData()
    sample_data, out_data = DataSplit(data)

    y = sample_data['diagnosis']
    y_out = out_data['diagnosis']
    x = sample_data.drop(columns='diagnosis')
    x_out = out_data.drop(columns='diagnosis')
    modelling.RunLogit(y,x,y_out,x_out)

    plotting.PlotScatters(y,x)
    plotting.PlotHists(x)
    plotting.PlotCorrelationMatrix(x, 'Feature Correlations')


def DataSplit(data):

    sample_data,out_data  = train_test_split(data, test_size=0.2)
    print(f'{len(sample_data)} training sample')
    print(f'{len(out_data)} test samples')

    return sample_data, out_data



if __name__ == '__main__':
    main()
    plt.show()