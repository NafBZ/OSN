from utils import *
from network import *

def train(X_train, X_test, y_train, y_test, model, path, save_model = False):
    X_train, X_test = prepare_inputs(X_train, X_test)
    y_train, y_test = prepare_targets(y_train, y_test)

    if model == 'svm':
        y_pred, y_prob = support_vector_machine(X_train, X_test, y_train, dir = path, save_model= save_model)

    elif model == 'tuned_svm':
        y_pred, y_prob = tuned_support_vector_machine(X_train, X_test, y_train, dir = path, save_model= save_model)

    elif model == 'logisticRegression':
        y_pred, y_prob = logistic_regression(X_train, X_test, y_train, dir = path, save_model= save_model)

    elif model == 'tuned_logisticRegression':
        y_pred, y_prob = tuned_logistic_regression(X_train, X_test, y_train, dir = path, save_model= save_model)

    evaluation_matrices(y_test,y_pred, y_prob)

    return