import argparse
import logging
import os
import time 
from datetime import datetime

from test import test
from data import Data_Handler
from train import train
from utils.setDir  import setDirectory

def str_to_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', nargs='?', default="./data/Nifti/In/Transformed/OCTA_SRL_256_V2", type=str, 
                        help='Enter the data path.(default: "./data/Nifti/In/FOV_66/SRL")')
                        # 2d - ./data/dataset/OG
                        # 2d - ./data/dataset/OG/OCTA_Enface
                        # 3d - ./data/Nifti/In/FOV_66/SRL - 640 but this gonna crop the volume to 256
                        # 3d before registration - ./data/Nifti/In/Transformed/OCTA_SRL_NO_REG
                        # 3d after registration - ./data/Nifti/In/Transformed/OCTA_SRL_256

    parser.add_argument('--label_path', nargs='?', default="./data/dataset/label/labels.xlsx", type=str, 
                        help='Enter the label path.(default: "./data/dataset/label/6mm/labels.xlsx")')

    parser.add_argument('--fold_num', nargs='?', default=5, type=int, 
                        help='Enter the # of folds.(default: 5)')

    parser.add_argument('--flatten', nargs='?', default='True', type=str_to_bool,
                        help='Flatten for 3D Volumes: True or False')

    parser.add_argument('--test_rate', nargs='?', default=0.15, type=float,
                        help='Set the rate of train/test : 0.1, 0.15, 0.2, 0.25, 0.3')

    parser.add_argument('--dimension', nargs='?', default='3d', type=str,
                        help='Choose the dimension between 2d/3d')

    parser.add_argument('--model', nargs='?', default='Res_10_3D', type=str,
                        help='Model name : VGG_16_2D, VGG_19_2D, \
                              Res_10_2D, Res_18_2D, Res_50_2D, VIT_2D,\
                              Incept_v3_2D, Google_2D,\
                              CV5FC2_3D, CV3FC2_3D,\
                              Res_10_3D, Res_18_3D, Res_50_3D, \
                              Incept_3D, Efficient_3D')

    parser.add_argument('--optimizer', nargs='?', default='asgd', type=str, 
                        help='Set the optimizer : sgd, asgd, adam, adagrad, adamW, adamp, adadelta, rmsp')

    parser.add_argument('--loss', nargs='?', default='nll', type=str,
                        help='Choose the loss function : ce, bce, mse, nll, fcl')

    parser.add_argument('--learningrate', nargs='?', default=5e-3, type=float, #5e-3 for ae_pre_train
                        help='Set the learning Rate')

    parser.add_argument('--epoch', nargs='?', default=200, type=int,
                        help='Set the epoch Number')

    parser.add_argument('--batch', nargs='?', default=1, type=int, #32 for ae_pre_train
                        help='Set the batch size')

    parser.add_argument('--disease', nargs='?', default="NORMAL AMD CSC DR RVO", type=str,
                        help='Group the disease that you want to utilize ex) "NORMAL AMD CNV CSC DR RVO OTHERS"')

    parser.add_argument('--mergeDisease', nargs='?', default='True', type=str_to_bool,
                        help='Set the disease to Abnormal')
    
    parser.add_argument('--filter', nargs='?', default='OG', type=str,
                        help='Using filter or not : OG / Curvelet')

    parser.add_argument('--layer', nargs='?', default='SRL', type=str,
                        help='Retinal layer Setting : SRL / DRL / Total(=OCTA)')

    parser.add_argument('--tfl', nargs='?', default='False', type=str_to_bool,
                        help='Set the transfer learning : True/False')

    parser.add_argument('--ae_pre_train', nargs='?', default='True', type=str_to_bool,
                        help='Autoencoder pre-traning : True/False')
    
    parser.add_argument('--weightDecay', nargs='?', default=0.15, type=float, #0.15
                        help='Setting the weight Decay')

    parser.add_argument('--tolerance', nargs='?', default=0.0, type=float, 
                        help='tolerance')

    parser.add_argument('--patience', nargs='?', default=15, type=int,  #30 for ae_pre_train
                        help='patience')

    parser.add_argument('--ae_data_num', nargs='?', default=500, type=int, 
                        help='ae_data_num')

    parser.add_argument('--transfer_learning_optimizer', nargs='?', default='adam', type=str, 
                        help='asgd, sgd, rmsp, adam, adamw, adadelta, adagrad') # asgd for ae_x

    parser.add_argument('--patch_mode', default='False', type=str_to_bool)
    
    parser.add_argument('--test_mode' , default='False', type=str_to_bool)
    
    parser.add_argument('--medcam', default='False', type=str_to_bool)
    
    parser.add_argument('--ae_learning_rate', nargs='?', default=5e-3, type=float) # 5e-3: loss 483-0.001607 # pre-train LR
    
    parser.add_argument('--num_class', nargs='?', default=2, type=int)

    args = parser.parse_args()
    args.date = datetime.today().strftime("%Y_%m%d")
    if 'res' in args.model.lower():
        args.res_depth = int(args.model.split('_')[1])
        
    return args

def main(args):
    # 1. Initiate DataHandler
    data_handler = Data_Handler(args)
    data_handler()

    logging.basicConfig(filename=f'{data_handler.getOuputDir()["log"]}/main.log', level=logging.INFO)
    logging.info(f"Argument : {args}")

    # 2. Training
    logging.info("Start train.")
    if not args.test_mode:
        train(args)(data_handler)
    
    # 3. Testing
    if args.ae_pre_train and args.dimension == '3d':
        return
    logging.info("Start testing")
    test(args)(data_handler)

if __name__ == "__main__": 
    args = getArguments()
    main(args)
