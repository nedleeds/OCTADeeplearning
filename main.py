import argparse
import logging
import os
import time 
from datetime import datetime

from test import test
from data import Data_Handler
from train import train

def str_to_bool(v):
    '''
    This function is used for convert the string input of the arguement parser to Boolean.
    Input string has ('yes', 'true', 't', 'y', '1'), then it will be True(bool).
    For the case of ('no', 'false', 'f', 'n', '0'), then it will be False(bool).
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', nargs='?', default="./data/Nifti/In/Transformed/OCTA_SRL_256", type=str, 
                        help='Enter the data path(default : "./data/Nifti/In/FOV_66/SRL").')
                        # 2d - ./data/dataset/OG - for patch mode
                        # 2d - ./data/dataset/OG/OCTA_Enface - Integrated FOV data
                        # 3d - ./data/Nifti/In/FOV_66/SRL - 640 but this gonna crop the volume to 256
                        # 3d before registration - ./data/Nifti/In/Transformed/OCTA_SRL_NO_REG
                        # 3d after registration - ./data/Nifti/In/Transformed/OCTA_SRL_256

    parser.add_argument('--label_path', nargs='?', default="./data/dataset/label/labels.xlsx", type=str, 
                        help='Enter the label path(default : "./data/dataset/label/6mm/labels.xlsx").')

    parser.add_argument('--fold_num', nargs='?', default=5, type=int, 
                        help='Enter the # of folds(default : 5).')

    parser.add_argument('--flatten', nargs='?', default='True', type=str_to_bool,
                        help='Flatten for 3D Volumes : True or False')

    parser.add_argument('--test_rate', nargs='?', default=0.15, type=float,
                        help='Set the rate of train/test : 0.1, 0.15, 0.2, 0.25, 0.3')

    parser.add_argument('--dimension', nargs='?', default='3d', type=str,
                        help='Choose the dimension between 2d/3d.')

    parser.add_argument('--model', nargs='?', default='Res_18_3D', type=str,
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

    parser.add_argument('--learningrate', nargs='?', default=3e-3, type=float, #5e-3 for ae_pre_train
                        help='Set the learning Rate.')

    parser.add_argument('--epoch', nargs='?', default=200, type=int,
                        help='Set the epoch Number.')

    parser.add_argument('--batch', nargs='?', default=1, type=int, #32 for ae_pre_train
                        help='Set the batch size.')

    parser.add_argument('--disease', nargs='?', default="NORMAL AMD CSC DR RVO", type=str,
                        help='Group the disease that you want to utilize ex) "NORMAL AMD CNV CSC DR RVO OTHERS".')

    parser.add_argument('--mergeDisease', nargs='?', default='True', type=str_to_bool,
                        help='Set the disease to Abnormal.')
    
    parser.add_argument('--filter', nargs='?', default='OG', type=str,
                        help='Using filter or not : OG / Curvelet.')

    parser.add_argument('--layer', nargs='?', default='SRL', type=str,
                        help='Retinal layer Setting : SRL / DRL / Total(=OCTA)')

    parser.add_argument('--tfl', nargs='?', default='False', type=str_to_bool,
                        help='Set the transfer learning : True/False')

    parser.add_argument('--ae_pre_train', nargs='?', default='False', type=str_to_bool,
                        help='Autoencoder pre-traning : True/False')
    
    parser.add_argument('--weightDecay', nargs='?', default=0.15, type=float, #0.15
                        help='Setting the weight Decay.')

    parser.add_argument('--tolerance', nargs='?', default=0.0, type=float, 
                        help='Set the tolerance for the EarlyStopping.')

    parser.add_argument('--patience', nargs='?', default=15, type=int,  #30 for ae_pre_train
                        help='Seth the patience for the EalryStopping.')

    parser.add_argument('--ae_data_num', nargs='?', default=500, type=int, 
                        help='Set the ae_data_num for autoencoder pre-training.')

    parser.add_argument('--transfer_learning_optimizer', nargs='?', default='adam', type=str, 
                        help='Set the optimizer for transfer learning.') # asgd for ae_x

    parser.add_argument('--patch_mode', default='False', type=str_to_bool,
                        help='The patch_mode is for 2D image. This will split the image to 4 parts.')
    
    parser.add_argument('--test_mode' , default='False', type=str_to_bool,
                        help='If test_mode is true, train will be skipped.')
    
    parser.add_argument('--medcam', default='True', type=str_to_bool,
                        help='This is for extracting 3D/2D attention map from Grad-CAM of M3D-CAM.')
    
    parser.add_argument('--ae_learning_rate', nargs='?', default=5e-3, type=float,
                        help='For Autoencoder pre-training learning rate.')
    
    parser.add_argument('--num_class', nargs='?', default=2, type=int,
                        help='Set the number of classes. 2 for binary classification n for selected diseases.')
    
    parser.add_argument('--clipping', default='True', type=str_to_bool,
                        help='Clipping before normalizing. If you use this, you can enhance the contrast of data.')

    args = parser.parse_args()
    
    # The date is save like 2022_0622. This can be used for save dir name.
    args.date = datetime.today().strftime("%Y_%m%d")
    
    # The depth of ResNet can be modified with this argument.
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
    args = get_arguments()
    main(args)
