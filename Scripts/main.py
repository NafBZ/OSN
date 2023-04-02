from dataloader import Dataset
from sklearn.model_selection import train_test_split
from utils import *
from network import *
from train import *

import yaml


with open("../config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

main_path = config['data']['root']
model = config['parameters']['model']
save_model = config['parameters']['save_model']
save_directory = config['data']['model_dir']

data = Dataset(path = main_path, train= True)
X_train, X_test, y_train, y_test = train_test_split(data.X,data.y, test_size = 0.20, random_state = 4, stratify = data.y)


train(X_train, X_test, y_train, y_test, model, save_directory, save_model= save_model)