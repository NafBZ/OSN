#Basic SVM
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle


def support_vector_machine(X_train, X_test, y_train, dir, model_name = 'svm', save_model = False, extension = '.sav'):

    svm = SVC(probability = True, kernel = 'linear')
    svm.fit(X_train, y_train)

    if save_model:
        with open(dir + model_name + extension, 'wb') as f:
            pickle.dump(svm, f)

    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:,1]


    return y_pred, y_prob


#Tuned SVM
def tuned_support_vector_machine(X_train, X_test, y_train, dir, model_name = 'tuned_svm', save_model = False, extension = '.sav'):
    # Instantiate the GridSearchCV object and run the search
    parameters = {'C':[0.1, 1, 10]}
    svm = SVC(kernel = 'rbf', probability = True)
    searcher = GridSearchCV(svm, parameters, cv = 5)
    searcher.fit(X_train, y_train)


    if save_model:
        with open(dir + model_name + extension, 'wb') as f:
            pickle.dump(searcher, f)

    y_pred = searcher.predict(X_test)
    y_prob = searcher.predict_proba(X_test)[:,1]

    # Report the best parameters and the corresponding score
    print("Best CV params", searcher.best_params_)
    print("Best CV accuracy", searcher.best_score_)

    return y_pred, y_prob


#Basic LogisticRegression
def logistic_regression(X_train, X_test, y_train, dir, model_name = 'logisticRegression', save_model = False, extension = '.sav'):

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    if save_model:
        with open(dir + model_name + extension, 'wb') as f:
            pickle.dump(logreg, f)

    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:,1]


    return y_pred, y_prob


#Tuned LogisticRegression
def tuned_logistic_regression(X_train, X_test, y_train, dir, model_name = 'tuned_logisticRegression', save_model = False, extension = '.sav'):
    c_space = np.logspace(-5, 8, 15)
    param_grid = {'solver' :['liblinear', 'saga']}
    
    logreg = LogisticRegression(penalty = 'l1')
    logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
    logreg_cv.fit(X_train, y_train)

    if save_model:
        with open(dir + model_name + extension, 'wb') as f:
            pickle.dump(logreg_cv, f)


    y_pred = logreg_cv.predict(X_test)
    y_prob= logreg_cv.predict_proba(X_test)[:,1]

    # Report the best parameters and the corresponding score
    print("Best CV params", logreg_cv.best_params_)
    print("Best CV accuracy", logreg_cv.best_score_)


    return y_pred, y_prob