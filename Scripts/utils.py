from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.metrics import roc_auc_score
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


