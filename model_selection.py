import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier

from nn_models import feedforward

def gen_estimator(model_generator,**sk_params):
    estimator = KerasClassifier(build_fn=model_generator,**sk_params)
    return estimator

def get_scores(estimator,x,y,score_funcs,score_func_params=None):
    if score_func_params is None:
        score_func_params = [{}]*len(score_funcs)
    y_pred = estimator.predict(x)
    return [ scorer(y,y_pred,**s_params) for scorer,s_params in zip(score_funcs,score_func_params)]

def multiscorer(estimator,x,y,score_funcs,score_func_params=None):
    scores = get_scores(estimator,x,y,score_funcs,score_func_params)
    print('score values')
    print(scores)
    return np.array(scores).sum()
    
def prepare_cvs(model,cv,scoring=[accuracy_score],n_splits=10,shuffle=True,**sk_params):
    cvs = {}
    est = KerasClassifier(build_fn=model,**sk_params)
    cvs['estimator'] = est
    # cvs['model'] = model(**sk_params)
    if len(scoring) > 1:
        cvs['scoring'] = multiscorer
    else:
        cvs['scoring'] = scoring[0]

    if cv == 'kfold':
        cvs['cv'] = KFold(n_splits=n_splits,shuffle=shuffle)
    elif cv in ['sss','stratified_shuffle_split']:
        cvs['cv'] = StratifiedShuffleSplit(n_splits=n_splits)

    return cvs

    
def cross_validation(X,Y,cv,model_generator,sample_weight=None,verbose=0):

    score_results=[]
    i=0
    for train_idx,test_idx in cv.split(X,Y):
        print('Split ',str(i+1))

        X_train,Y_train = X[train_idx],Y[train_idx]
        X_test,Y_test = X[test_idx],Y[test_idx]
        # Instantiate model
        m = model_generator(layer_spec=[128],num_labels=Y.shape[-1],
                                optim='adam',reg_weight=0.01)

        # Fit model
        m.fit(X_train,Y_train,
              epochs=500,verbose=verbose, batch_size=32,
              sample_weight=sample_weight[train_idx],
              validation_split=0.1,
              )

        # Score model
        score = m.evaluate(X_test,Y_test,batch_size=128,verbose=0)
        print('Split ',str(i+1),' Score: ',score)
        score_results.append(score)
        i+=1

    return score_results