import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedShuffleSplit,KFold,LeaveOneGroupOut
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import accuracy_score,make_scorer,precision_recall_fscore_support
from keras.wrappers.scikit_learn import KerasClassifier

from nn_models import feedforward,basic_rnn
from utils.io import get_oh_labels

class CVScore():

    def __init__(self,estimator,cv=None,score_func=precision_recall_fscore_support):

        self.estimator = estimator
        if cv is not None:
            self.cv = cv
        else:
            self.cv = KFold()

        self.score_func = score_func
        self.train_idxs = []
        self.test_idxs = []
        self.scoring_metrics = ['accuracy','precision','recall','fscore']
        self.cv_results_ = {}
        self.models_ = []

        for m in self.scoring_metrics:
            self.cv_results_['test_'+m]=[]

    def save_results(self,fp):
        cv_df = pd.DataFrame(self.cv_results_)
        print('Writing results '+fp+'...')
        cv_df.to_pickle(fp)

    def save_parameters(self,fp,fit_params,model_params):
        params = dict(
            fit=fit_params,
            model=model_params,
        )
        print('Attempting to save params at '+fp+'...')
        pickle.dumps(params,open(fp,'wb'))  

    def fit(self,X,y,cv_iter,oh_encoder,fit_params):

        self.y_classes = np.unique(y)
        sample_weight = fit_params.pop('sample_weight',None)

        i = 1
        for train_idx,test_idx in cv_iter:
            x_train,y_train = X[train_idx],y[train_idx]
            x_test,y_true = X[test_idx],y[test_idx]
            if sample_weight is not None:
                train_sample_weight = sample_weight[train_idx]
            else:
                train_sample_weight = None

            y_train_oh = get_oh_labels(y_train,oh_encoder)

            self.estimator.fit(x_train,y_train_oh,sample_weight=train_sample_weight,
                **fit_params)

            l,a = self.estimator.model.evaluate(x_test,get_oh_labels(y_true,oh_encoder),batch_size=128,verbose=0)
            y_pred_class=self.estimator.model.predict_classes(x_test,batch_size=128,verbose=0) 
            y_pred = [ self.y_classes[v] for v in y_pred_class]
            p,r,f,_ = self.score_func(y_true,y_pred,average='weighted')
            print('Split %d scores: '%i,[a,p,r,f])
            cols = ['test_'+s for s in self.scoring_metrics]
            for k,v in zip(cols,[a,p,r,f]):
                self.cv_results_[k].append(v)
            self.models_.append(self.estimator.model)
            self.train_idxs.append(train_idx)
            self.test_idxs.append(test_idx)
            i+=1

def prepare_cvs(model,cv,scoring=[accuracy_score],n_splits=9,shuffle=True,groups=None,**sk_params):
    models = dict(
        feedforward=feedforward,
        basic_rnn=basic_rnn
    )
    scoring = dict(
        prfs=precision_recall_fscore_support,
        acc=accuracy_score
    )
    cvs = {}
    est = KerasClassifier(build_fn=models[model],**sk_params)
    cvs['estimator'] = est
    # cvs['model'] = model(**sk_params)

    if cv == 'kfold':
        cvs['cv'] = KFold(n_splits=n_splits,shuffle=shuffle)
    elif cv in ['sss','stratified_shuffle_split']:
        cvs['cv'] = StratifiedShuffleSplit(n_splits=n_splits)
    elif cv in ['logo','LeaveOneGroupOut']:
        cvs['cv'] = LeaveOneGroupOut()

    return cvs
