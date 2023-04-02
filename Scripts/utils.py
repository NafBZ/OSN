from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

root = '../model/'

def prepare_inputs(X_train, X_test, model_name = 'input_encoder.sav' , dir = root):
        
    #create an instance of Ordinal Encoder
    oe = OrdinalEncoder()

    #fit the training data
    oe.fit(X_train)

    #save the model for future purposes
    with open(dir + model_name, 'wb') as f:
        pickle.dump(oe, f)

    #prepare the encoded data
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)

    return X_train_enc, X_test_enc


def prepare_targets(y_train, y_test, model_name = 'label_encoder.sav'):

    #create an instance of Ordinal Encoder
    le = LabelEncoder()

    #fit the labels
    le.fit(y_train)

    #save the model for future purposes
    with open(dir + model_name, 'wb') as f:
        pickle.dump(le, f)

    #prepare the labels
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    return y_train_enc, y_test_enc

def evaluation_matrices(test, predictions, probability):
    roc_auc = roc_auc_score(test, probability)

    print(f'The evaluation based on Precision, Recall etc: \n {classification_report(test,predictions)}')
    print(f'The confusion matrix is: \n {confusion_matrix(test, predictions)}')
    print(f'The log loss of the model is: {log_loss(test,probability)}')
    print(f'AUC of ROC Curve: {roc_auc}')


#Basic SVM
def support_vector_machine(X_train, X_test, y_train, y_test):

    svm = SVC(probability = True, kernel = 'linear')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:,1]


    return y_pred, y_prob


#Tuned SVM
def tuned_support_vector_machine(X_train, X_test, y_train, y_test):
    # Instantiate the GridSearchCV object and run the search
    parameters = {'C':[0.1, 1, 10]}
    svm = SVC(kernel = 'rbf', probability = True)
    searcher = GridSearchCV(svm, parameters, cv = 5)
    searcher.fit(X_train, y_train)

    y_pred = searcher.predict(X_test)
    y_prob = searcher.predict_proba(X_test)[:,1]

    # Report the best parameters and the corresponding score
    print("Best CV params", searcher.best_params_)
    print("Best CV accuracy", searcher.best_score_)

    # Report the test accuracy using these best parameters
    print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

    return y_pred, y_prob