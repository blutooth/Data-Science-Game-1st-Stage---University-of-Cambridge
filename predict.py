import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from models import *
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import *
from sklearn.ensemble import RandomForestClassifier

def build_X_meta(model, X):
    preds = model.predict(X, batch_size=64)

    rotation_preds = [np.copy(preds)]
    rotate90_avg(X, preds, model, rotation_preds)

    Xm = np.array(rotation_preds).transpose((1,0,2)).reshape(X.shape[0],16)


    X_a = X.copy()
    for i in range(len(X)):
        X_a[i] = np.fliplr(X_a[i])

    this_preds = model.predict(X_a, batch_size=350)
    preds += this_preds
    rotation_preds = [np.copy(this_preds)]
    rotate90_avg(X_a, preds, model, rotation_preds)
    Xm = np.hstack((Xm, np.array(rotation_preds).transpose((1,0,2)).reshape(X.shape[0],16)))


    X_a = X.copy()
    for i in range(len(X)):
        X_a[i] = np.flipud(X_a[i])

    this_preds = model.predict(X_a, batch_size=350)
    preds += this_preds
    rotation_preds = [np.copy(this_preds)]
    rotate90_avg(X_a, preds, model, rotation_preds)
    Xm = np.hstack((Xm, np.array(rotation_preds).transpose((1,0,2)).reshape(X.shape[0],16)))

    for i in range(len(X)):
        X_a[i] = np.fliplr(X_a[i])

    this_preds = model.predict(X_a, batch_size=350)
    preds += this_preds
    rotation_preds = [np.copy(this_preds)]
    rotate90_avg(X_a, preds, model, rotation_preds)
    Xm = np.hstack((Xm, np.array(rotation_preds).transpose((1,0,2)).reshape(X.shape[0],16)))
    return Xm


model, model_name = get_model()
model.load_weights(model_name+'0.86')

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y = np.load(os.path.join(DATA_DIR, 'y_train.npy')) -1
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
test_ids = np.load(os.path.join(DATA_DIR, 'test_ids.npy'))


Xm_train = build_X_meta(model, X_train)
Xm_test = build_X_meta(model, X_test)






RF = RandomForestClassifier(n_estimators=600, n_jobs=-1, random_state=0)
RF.fit(Xm_train, y)

preds = RF.predict_proba(Xm_test)

preds = np.argmax(preds, axis=1)

write_subm(preds, test_ids, 'submission.csv')

