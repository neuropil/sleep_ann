import argparse
from pprint import pprint as pp

def gen_arg_parser():
    parser = argparse.ArgumentParser(description='Helper function for doing model evaluation')
    parser.add_argument('network',type=str,choices=['feedforward','rnn'],default='feedforward',
        help='Architecture to evaluate')

    # Add evaluation type group
    mode_type = parser.add_mutually_exclusive_group(required=True)
    mode_type.add_argument('-i','--individual',action='store_const',dest='mode',const='individual')
    mode_type.add_argument('-c','--combined',action='store_const',dest='mode',const='individual')
    mode_type.add_argument('-l','--leaveout',action='store_const',dest='mode',const='individual')

    # Add split type group
    split_type = parser.add_mutually_exclusive_group(required=True)
    split_type.add_argument('--kfold',action='store_const',dest='split_type',const='kfold')
    split_type.add_argument('-s','--sss',action='store_const',dest='split_type',const='sss',
        help='Split using Stratified Shuffle Split')

    parser.add_argument('-n','--n_splits',type=int,action='store',default=10)
    return parser

if __name__ == "__main__":
    parser = gen_arg_parser()
    args = parser.parse_args()
    pp(vars(args))
    # import ipdb; ipdb.set_trace()