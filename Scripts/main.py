from dataloader import Dataset
from sklearn.model_selection import train_test_split
from utils import *
from network import *

df = Dataset(train= True)


X_train, X_test, y_train, y_test = train_test_split(df.X,df.y, test_size = 0.20, random_state = 4, stratify = df.y)
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

support_vector_machine(X_train_enc, X_test_enc, y_train_enc, y_test_enc)