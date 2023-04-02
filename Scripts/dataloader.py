# prepare input data
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, path = '../DataSet/', train = False):
        self.path = path

        if train:
            train_data = os.path.join(path,'train.csv')
            self.data = pd.read_csv(train_data)
            self.X = self.data.drop(['CLASS', 'LIQUOR ITEM'], axis = 1)
        else:
            test_data = os.path.join(path, 'test.csv')
            self.data = pd.read_csv(test_data)
            self.X = self.data.drop(['CLASS', 'LIQUOR ITEM', 'Class'], axis = 1)
        
        self.y = self.data.CLASS


    def data_attributes(self):
        rows = len(self.data)
        columns = len(self.data.columns)

        print(f'Numberof rows are {rows} \n')
        print(f'Numberof columns are {columns} \n')


    def class_visualisation(self):

        sns.set_style('whitegrid')
        sns.set_context('paper')

        fig = sns.catplot(x = 'CLASS' , data = self.data, kind = 'count', height = 4.5, aspect = 1 )
        fig.set(xlabel = 'Classes', ylabel ='Samples in each class')
        plt.show()



    