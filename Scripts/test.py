import pickle
import os
from utils import *
from dataloader import *
import yaml
from sklearn.model_selection import train_test_split
import numpy as np


def test(X_train, X_test, y_train, y_test, path, filename):


    model_path = os.path.join(path, filename)
    loaded_model = pickle.load(open(model_path, 'rb'))
    X_train, X_test = prepare_inputs(X_train, X_test)
    y_train, y_test = prepare_targets(y_train, y_test)

    test_features = np.concatenate((X_train, X_test))
    test_labels = np.concatenate([y_train, y_test])

    y_pred = loaded_model.predict(test_features)
    y_prob = loaded_model.predict_proba(test_features)[:,1]

    evaluation_matrices(test_labels,y_pred, y_prob)


if __name__ == '__main__':


    with open("../config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)

    main_path = config['data']['root']
    save_directory = config['data']['model_dir']

    data = Dataset(path = main_path)
    X_train, X_test, y_train, y_test = train_test_split(data.X,data.y, test_size = 0.20, random_state = 4, stratify = data.y)
    test(X_train, X_test, y_train, y_test, path=save_directory, filename='svm.sav')
