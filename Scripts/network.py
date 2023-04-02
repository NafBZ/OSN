#Basic SVM
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np


def support_vector_machine(X_train, X_test, y_train):

    svm = SVC(probability = True, kernel = 'linear')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:,1]


    return y_pred, y_prob


#Tuned SVM
def tuned_support_vector_machine(X_train, X_test, y_train):
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

    return y_pred, y_prob


#Basic LogisticRegression
def logistic_regression(X_train, X_test, y_train, y_test):

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:,1]


    return y_pred, y_prob


#Tuned LogisticRegression
def logreg_func_tuned(X_train, X_test, y_train, y_test):
    c_space = np.logspace(-5, 8, 15)
    param_grid = {'solver' :['liblinear', 'saga']}
    
    logreg = LogisticRegression(penalty = 'l1')
    logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
    logreg_cv.fit(X_train, y_train)
    y_pred = logreg_cv.predict(X_test)
    y_prob= logreg_cv.predict_proba(X_test)[:,1]

    # Report the best parameters and the corresponding score
    print("Best CV params", logreg_cv.best_params_)
    print("Best CV accuracy", logreg_cv.best_score_)


    return y_pred, y_prob