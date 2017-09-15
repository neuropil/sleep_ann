import argparse
import numpy as np
from pprint import pprint as pp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from nn_models import feedforward,basic_rnn
from model_selection import CVScore,prepare_cvs
from utils import load_preprocessed,get_pow_bands,get_inverse_freq_weights

def gen_arg_parser():
    parser = argparse.ArgumentParser(description='Helper function for doing model evaluation')
    parser.add_argument('network',type=str,choices=['feedforward','basic_rnn'],default=feedforward,
        help='Architecture to evaluate')

    model_args=parser.add_argument_group(title='model_params')
    model_args.add_argument('--optimizer',type=str,choices=['adam','nadam','rmsprop','sgd'],default='adam')
    model_args.add_argument('--kernel_reg_weight',type=float,default=0.01)
    model_args.add_argument('--activation',type=str,choices=['sigmoid','relu','tanh'],default='sigmoid')
    model_args.add_argument('--loss',type=str,default='categorical_crossentropy')
    model_args.add_argument('--num_labels',type=int,default=None)

    fit_args=parser.add_argument_group(title='training parameters')
    fit_args.add_argument('--epochs',type=int,default=10)
    fit_args.add_argument('--scoring', type=str, action='append',choices=['acc','prf'],
        default='acc')
    fit_args.add_argument('--sample_weight',type=str,choices=['inverse_frequency','sq_inv_freq'],default=None)

    # Add evaluation type group
    mode_type = parser.add_argument_group(title='evaluation args')
    mode_type.add_argument('-i','--individual',action='store_const',dest='mode',const='individual',default='merged')
    mode_type.add_argument('-p','--data_dir_root',type=str, default='/home/elijahc/data/uminn/preprocessed')
    mode_type.add_argument('--no_scaling',action='store_false', dest='scaler',default='True')
    mode_type.add_argument('--merge_nrem',action='store_true', dest='merge_nrem',default=False)

    # Add cross-validation group
    split_type = parser.add_mutually_exclusive_group(required=True)
    split_type.add_argument('--kfold',action='store_const',dest='cv_type',const='kfold')
    split_type.add_argument('-s','--sss',action='store_const',dest='cv_type',const='sss',
        help='Split using Stratified Shuffle Split')
    split_type.add_argument('--logo',action='store_const',dest='cv_type',const='logo',
        help='LeaveOneGroupOut')

    parser.add_argument('-n','--n_splits',type=int,action='store',default=10)

    return parser

if __name__ == "__main__":
    scaler=None
    parser = gen_arg_parser()
    args = parser.parse_args()
    merged_data = load_preprocessed(args.data_dir_root,simple=args.merge_nrem,merge_keys=['stages','pows'])
    if args.scaler:
        scaler=StandardScaler()
        scaler.fit(merged_data['pows'])
    if args.mode == 'merged':
        data = merged_data
    else:
        # Implement this
        pass

    if args.merge_nrem==True:
        sleep_stages = data['stages_simple']
    else:
        sleep_stages = data['stages'][:,2].astype(np.int8)

    if args.num_labels==None:
        num_labels=len(np.unique(sleep_stages))
    model_params = dict(
        activ=args.activation,
        reg_weight=args.kernel_reg_weight,
        optim=args.optimizer,
        num_labels=args.num_labels,
        loss=args.loss,
    )
    fit_params = {}
    fit_params['epochs']=args.epochs
    fit_params['verbose']=0
    cvs = prepare_cvs(model=args.network,cv=args.cv_type,scoring=args.scoring,n_splits=args.n_splits,
        **model_params,
    )
    pp(vars(args))
    power_bands = get_pow_bands(data,scaler)
    fit_params['sample_weight'] = get_inverse_freq_weights(sleep_stages)
    import ipdb; ipdb.set_trace()
    clf = CVScore(**cvs)
    clf.fit(X=power_bands,y=sleep_stages,fit_params=fit_params)