import os
import random
import copy
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import torch
import torch.nn as nn
import torchsummary

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch_3d import EfficientNet3D

from utils.resnet import generate_model
from utils.evaluate import checking
from utils.FocalLoss import FocalLoss
from utils.earlyStop import EarlyStopping
from utils.INCEPT_V3_3D import Inception3_3D

from model import ResNet_2D, GOOGLE_2D, INCEPT_V3_2D, VGG_2D, EFFICIENT_2D, VIT_2D
from model import CV3FC2_3D, CV5FC2_3D, VGG16_3D, autoencoder, freeze

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ## Totally for reproducibility.
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    # torch.backends.cudnn.benchmark = True

class train():
    def __init__(self, args):
        self.args = args
        self.is_patch       = args.patch_mode
        self.model_name     = args.model
        self.optimizer_name = args.optimizer.lower()
        self.loss_name      = args.loss.lower()
        self.lr             = args.learningrate
        self.epoch          = args.epoch
        self.batch          = args.batch
        self.fold_num       = args.fold_num
        self.is_transfered  = args.tfl
        self.wd             = args.weightDecay
        self.tolerance      = args.tolerance
        self.patience       = args.patience
        self.ae_data_num    = args.ae_data_num
        self.isMerge        = args.mergeDisease
        self.filter         = args.filter
        self.dimension      = args.dimension
        self.classes        = args.num_class
        self.ae_pre_train   = args.ae_pre_train
        self.best_epoch     = 1
        self.valid_subsampler = []
        self.log_dir        = ''
        self.check_dir      = ''
        self.result_dir     = ''
        self.best_param_dir = ''
        self.tensorboard_dir= ''
        self.pre_writer     = None
        self.tf_lrn_opt     = args.transfer_learning_optimizer
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, data_handler):
        self.set_output_dir(data_handler)
        self.set_train_data(data_handler)
        
        self.fold_Best       = [0]*self.fold_num
        self.previous_BEST   = [-np.inf for _ in range(self.fold_num)]
        self._best_parameter = {f'fold{idx}':{} for idx in range(1, self.fold_num+1)}
        self.y_folds         = []
        self.y_preds_folds   = []

        self.pre_trained_batch = 1
        self.pre_trained_lr    = self.args.ae_learning_rate
        self.classes           = len(data_handler.get_disease_keys())

        # checking model
        logging.info(f"Status-{list(set(self.index_disease.values()))}")

        if self.ae_pre_train and ('3' in self.dimension):
            seed_everything(99) ## 5 for my AE
            ## using total set
            logging.info("Start AutoEncoder pre-processing.")
            self.ae_pre_train(data_handler)
            # self.test_ae_pre_train(data_handler)
            return 
            # del totalset, totalloader
        
        # do stratified k-fold
        logging.info("Start stratified K-Fold Cross Validation.")
        self.total_Best_cm = [0]*self.fold_num

        if self.isMerge:
            self.classes = 2
            self.binary_classification(data_handler)
            # self.retrain(data_handler) 
        else:
            self.multi_classification(data_handler)
            # self.retrain(data_handler) 

    def set_output_dir(self, data_handler):
        '''
        Set the output directories' path.
        '''
        ae = 'ae_o' if self.is_transfered else 'ae_x'
        self.best_param_dir = os.path.join(self.best_param_dir, ae)
        self.log_dir        = data_handler.getOuputDir()['log']
        self.check_dir      = data_handler.getOuputDir()['checkpoint']
        self.result_dir     = data_handler.getOuputDir()['result']
        self.best_param_dir = os.path.join(data_handler.getOuputDir()['best_parameter'], ae)
        self.tensorboard_dir= os.path.join(data_handler.getOuputDir()['tensorboard'], ae)
        self.roc_plot_dir= os.path.join(data_handler.getOuputDir()['roc_plot'], ae)

    def set_train_data(self, data_handler):
        '''
        Set the data features.
        '''
        self.input_shape   = data_handler.getInputShape()
        self.disease_index = data_handler.sort_table(reverse=False)
        self.index_disease = data_handler.sort_table(reverse=True)
        self.label_table   = [self.disease_index, self.index_disease]

    def multi_classification(self, data_handler):
        for fold_idx in range(1, self.fold_num+1):
            seed_everything(34)
            data_handler.set_dataset('train', fold_idx=fold_idx)
            data_handler.set_dataset('valid', fold_idx=fold_idx)
            self.input_shape = data_handler.getInputShape()
            self.classes = len(data_handler.getDiseaseLabel())
            self.current_disease = sorted(list(data_handler.getDiseaseLabel().keys()))
            check = checking(lss=self.loss_name, labels=data_handler.getDiseaseLabel(), isMerge=self.isMerge)
            class_weights = 1./torch.tensor(self.classes, dtype=torch.float) 
            cnt = 0
            digging = False
            while not digging:
                # settings
                cnt += 1

                dig_score = {'1':0.80,#87, 
                        '2':0.80,#87, 
                        '3':0.80,#85, 
                        '4':0.79,#84, 
                        '5':0.80}#83} # seed 33

                self.previous_BEST[fold_idx-1] = 0.
                self._gamma = 0.94

                # set the dataset
                dataset = { 'train' : iter(data_handler.gety()['train']) , 
                            'valid' : iter(data_handler.gety()['valid']) }
                dataset_sizes = self.printDataNum(fold_idx, dataset)
                if self.is_transfered and ('3' in self.dimension):
                    # loadModel <- load Best model from AE preTrained.
                    # self.loss_name = 'ce'
                    # self.optimizer_name = 'asgd'
                    model = next(self.loadModel("autoencoder"))
                    model = freeze(num_class=self.classes, model=model).to(self.device)
                    self.lr = self.args.learningrate # learning rate for transfer learning 
                    self.optimizer_name = self.tf_lrn_opt         
                    #sgd, asgd, rmsp, adam, adamw, adagrad, adadelta
                else : 
                    model = next(self.getModel()).to(self.device)
                    # freezing Convolution layers and set FC layer's required_grad True.
                    if '2' in self.dimension:
                        if 'vgg' in self.model_name:
                            for i,l in enumerate(model.vgg16.features):
                                if len(model.vgg16.features)-5<i<len(model.vgg16.features):
                                    l.requires_grad = True
                                else:
                                    l.requires_grad = False

                self.loss_name = self.args.loss.lower()
                best_model_wts = copy.deepcopy(model.state_dict())
                best_train = -9999
                worse_cnt = 0
                previous_loss = 9999.
                epoch = 1
                metrics = {'train':None, 'valid':None}
                earlyStop = EarlyStopping(key={'name':'Loss','mode':'min'}, tolerance=self.tolerance, patience=self.patience)
                best_epoch_gt = []
                best_epoch_pd = []
                lss_class = next(self.getLoss())
                if torch.cuda.device_count() > 1 and '50' in self.model_name:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)     
                
                optimizer = next(self.getOptimizer(model.parameters(), lr=self.lr))
                scheduler = StepLR(optimizer, step_size=10, gamma=self._gamma)
                while not earlyStop.step(metrics['valid']):
                    writer_remove = True if epoch == 1 else False
                    writer = self.initWriter(fold_idx, f'{self.fold_num}fold',cnt, writer_remove)
                    print('-'*46+f'\nEpoch {epoch}/{self.epoch} - cnt[{cnt}], dig_score[{dig_score[f"{fold_idx}"]}]')
                    # metrics = {'train':None, 'valid':None}
                    model.zero_grad()
                    for phase in ['train', 'valid']:
                        data_handler.set_phase(phase)
                        epoch_gt, epoch_pd = [], []
                        if phase == 'train' : 
                            model.train()  # 모델을 학습 모드로 설정
                        else : 
                            model.eval()   # 모델을 평가 모드로 설정
                        epoch_loss = 0.0
                        
                        for step, (train_X, train_y) in enumerate(DataLoader(data_handler, batch_size=self.batch, shuffle=True)):
                            train_X = train_X[0] if '2' in self.dimension else train_X[0].unsqueeze_(1)
                            train_y = train_y[0].long()
                            # print(f'step:{step}-{train_y}')
                            # 매개변수 경사도를 0으로 설정
                            optimizer.zero_grad()
                            # 순전파
                            # 학습 시에만 연산 기록을 추적
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(train_X)
                                
                                if 'incept' in self.model_name.lower():
                                    if '3' in self.dimension:
                                        if phase =='train':
                                            prediction = next(self.doActivation(outputs[0]))
                                        else:
                                            prediction = next(self.doActivation(outputs))
                                    else:
                                        prediction = next(self.doActivation(outputs))
                                else:
                                    prediction = next(self.doActivation(outputs))
                                    
                                _, preds = torch.max(prediction, 1)
                                loss = lss_class(prediction, train_y)
                                
                                step_pd = preds.data.cpu().numpy()
                                step_gt = train_y.data.cpu().numpy()
                                
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                            
                            epoch_gt.extend(step_gt)
                            epoch_pd.extend(step_pd)
                            step_loss = loss.item()
                            epoch_loss += step_loss*len(train_y)
                            del train_X, train_y, prediction
                            
                        if phase == 'train':
                            scheduler.step()
                            
                        epoch_loss_mean = round(epoch_loss/dataset_sizes[phase], 6)
                        
                        print(f'{"="*38}{phase}{"="*38}')
                        metric, cfmx = check.Epoch(epoch_gt, epoch_pd)
                        metrics[phase] = metric
                        metrics[phase]['Loss'] = epoch_loss_mean
                        thsh_key = 'f1'
                        if 'acc' in thsh_key:
                            metric_for_thresh = metric[thsh_key]
                        elif 'f1' in thsh_key:
                            metric_for_thresh = metric['macro avg']['f1-score']
                            
                        self.saveResult(epoch, phase, fold_idx, epoch_loss_mean, metric, cfmx)
                        
                        if phase=='train' :
                            epoch_loss_train=epoch_loss_mean
                            if best_train <= metric_for_thresh:
                                best_train = metric_for_thresh
                        else: 
                            epoch_loss_valid = epoch_loss_mean
                            if self.previous_BEST[fold_idx-1]<= metric_for_thresh:
                                print(f'epoch[{epoch}] is best {thsh_key}')
                                self.saveResult(epoch, 'best', fold_idx, epoch_loss_valid, metrics[phase], cfmx)
                                best_epoch = epoch
                                best_loss = epoch_loss_valid
                                best_model_wts = copy.deepcopy(model.state_dict())
                                self.previous_BEST[fold_idx-1] = metric_for_thresh
                                self.total_Best_cm[fold_idx-1] = metric_for_thresh

                            if previous_loss >= epoch_loss_valid:
                                previous_loss = epoch_loss_valid
                            else:
                                worse_cnt+=1

                            writer.add_scalars('Loss',{'train':epoch_loss_train, 'valid':epoch_loss_valid}, epoch)
                            writer.add_scalars('ACC', {'train':metrics['train']['accuracy'],'valid':metrics['valid']['accuracy']}, epoch)
                    epoch += 1
                    writer.close()

                if self.previous_BEST[fold_idx-1]>=dig_score[f'{fold_idx}']:
                    self.y_folds.extend(epoch_gt)
                    self.y_preds_folds.extend(epoch_pd)
                    self.saveModel('best', fold_idx, best_epoch, best_model_wts, best_loss)
                    digging = True
                    break

    def binary_classification(self, data_handler):
        '''
        This function is for the binary-classification(Normal/Abnormal).
        Basically, train step is based on Pytorch-classification code.
        'checking' module is for the evalutation.
        '''
        for fold_idx in range(1, self.fold_num+1):
            seed_everything(34)
            check = checking(lss=self.loss_name, labels=self.label_table, isMerge=self.isMerge)
            data_handler.set_dataset('train', fold_idx=fold_idx)
            data_handler.set_dataset('valid', fold_idx=fold_idx)
            self.input_shape = data_handler.getInputShape()
            cnt = 0
            dig = True
            while dig:
                self.previous_BEST[fold_idx-1] = 0.
                self._gamma = 0.94
                # settings
                cnt += 1
                best_train = -9999
                worse_cnt = 0
                previous_loss = 9999.
                epoch = 1
                metrics = {'train':None, 'valid':None}
                key_metric = 'F1'
                earlyStop = EarlyStopping(key={'name':'Loss','mode':'min'}, 
                                          tolerance=self.tolerance, 
                                          patience=self.patience)
                
                dig_score = {'1':0.80, '2':0.80, '3':0.80, '4':0.80, '5':0.80}
                # dig_score = {'1':0.87-cnt%100, '2': 0.87-cnt%100, '3':0.87-cnt%100, '4':0.84-cnt%100, '5':0.87-cnt%100}
                
                # set the dataset
                dataset = { 'train' : iter(data_handler.gety()['train']) , 
                            'valid' : iter(data_handler.gety()['valid']) }
                dataset_sizes = self.printDataNum(fold_idx, dataset)
                
                if self.is_transfered and ('3' in self.dimension):
                    '''Mode : 3D Transfer Learning(using Autoencoder)'''
                    # load pre-trained model.
                    self.loss_name = 'nll'
                    self.optimizer_name = 'asgd' 
                    model = next(self.loadModel("autoencoder"))
                    model = freeze(num_class=self.classes, model=model).to(self.device)
                    
                    # set parameters for fine-tunning of transfer learning.
                    self.lr = self.args.learningrate
                    self.optimizer_name = self.tf_lrn_opt
                    self.loss_name = self.args.loss
                else:
                    # init model
                    model = next(self.getModel())
                    # freezing for fine-tunning
                    if '2' in self.dimension:
                        if 'vgg' in self.model_name:
                            for i, l in enumerate(model.vgg16.features):
                                if len(model.vgg16.features)-5<i<len(model.vgg16.features):
                                    l.requires_grad = True
                                else:
                                    l.requires_grad = False
                
                optimizer = next(self.getOptimizer(model.parameters(), lr=self.lr))
                lss_class = next(self.getLoss())
                scheduler = StepLR(optimizer, step_size=10, gamma=self._gamma)
                best_model_wts = copy.deepcopy(model.state_dict())
                
                if torch.cuda.device_count() > 1 and '50' in self.model_name:
                    '''
                    You don't need to use this if you don't have multiple gpus 
                    which has same spec. However, as this model is huge,
                    'Dataparallel' is highly recommended for ResNet3D-50.
                    '''
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
                    
                while not earlyStop.step(metrics['valid']):
                    '''
                    This is for the EarlyStop.
                    Basically this is working on the validation loss.
                    '''
                    writer_remove = True if epoch == 1 else False
                    writer = self.initWriter(fold_idx, f'{self.fold_num}fold', cnt, writer_remove)
                    
                    print('-'*46+f'\nEpoch {epoch}/{self.epoch} - cnt[{cnt}], thsh[{dig_score[f"{fold_idx}"]}]')
                    model.zero_grad()
                    for phase in ['train', 'valid']:
                        epoch_gt, epoch_pd = [], []                         
                        epoch_loss = 0.0
                        data_handler.set_phase(phase)
                        model.train() if phase == 'train' else model.eval()
                        for step, (train_X, train_y) in enumerate(DataLoader(data_handler, 
                                                                             batch_size=self.batch, 
                                                                             shuffle=True)):
                            train_X = train_X[0] if '2' in self.dimension else train_X[0].unsqueeze_(1).to(self.device)
                            train_y = train_y[0].long().to(self.device)
                            optimizer.zero_grad()
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(train_X)
                                prediction = next(self.doActivation(outputs))
                                _, preds = torch.max(prediction, 1)
                                loss = lss_class(prediction, train_y)
                                step_pd = preds.data.cpu().numpy()
                                step_gt = train_y.data.cpu().numpy()
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                            epoch_gt.extend(step_gt)
                            epoch_pd.extend(step_pd)
                            check.Step(step_gt, step_pd)
                            step_loss = loss.item()
                            epoch_loss += step_loss*len(train_y)
                            del train_X, train_y, prediction
                        
                        if phase == 'train':
                            scheduler.step()
                        epoch_loss_mean = epoch_loss/dataset_sizes[phase]
                        
                        print(f'{phase} Loss : {round(epoch_loss_mean, 6)}', end=' ')
                        metric, _  = check.Epoch(epoch_gt, epoch_pd)
                        # _ : confusion matrix
                        metrics[phase] = metric
                        metrics[phase]['Loss'] = epoch_loss_mean
                        # save train scores
                        self.saveResult(epoch, phase, fold_idx, epoch_loss_mean, metric)
                        if phase == 'train': 
                           epoch_loss_train=epoch_loss_mean
                           if best_train <= metrics[phase][key_metric]:
                                best_train = metrics[phase][key_metric]
                        else: 
                            epoch_loss_valid = epoch_loss_mean
                            if self.previous_BEST[fold_idx-1]<= metrics[phase][key_metric]:
                                print(f'epoch[{epoch}] is best {key_metric}')
                                self.saveResult(epoch, 'best', fold_idx, epoch_loss_valid, metric)
                                best_epoch = epoch
                                best_loss = epoch_loss_valid
                                best_model_wts = copy.deepcopy(model.state_dict())
                                self.previous_BEST[fold_idx-1] = metrics[phase][key_metric]
                                self.total_Best_cm[fold_idx-1] = metrics[phase]
                            if previous_loss >= epoch_loss_valid:
                                previous_loss = epoch_loss_valid
                            else:
                                worse_cnt+=1
                            writer.add_scalars('Loss',{'train':epoch_loss_train, 'valid':epoch_loss_valid}, epoch)
                            writer.add_scalars('ACC', {'train':metrics['train']['ACC'],'valid':metrics['valid']['ACC']}, epoch)
                            writer.add_scalars('F1',  {'train':metrics['train']['F1'], 'valid':metrics['valid']['F1'] }, epoch)
                    epoch += 1
                    writer.close()
                    
                if self.previous_BEST[fold_idx-1]>=dig_score[f'{fold_idx}']:
                    self.y_folds.extend(epoch_gt)
                    self.y_preds_folds.extend(epoch_pd)
                    self.saveModel('best', fold_idx, best_epoch, best_model_wts, best_loss)
                    dig = False
                    break
   
    def ae_pre_train(self, data_handler):
        '''
        This function is for the autoencoder transfer learning.
        '''
        self.lr = self.pre_trained_lr
        self.batch = self.pre_trained_batch
        # lr = 0.001(default) -> 0.0003 -> 0.003 -> 0.01 -> +scheduler
        # data_handler.set_dataset('train', data_num= 130 if self.ae_data_num == 500 else None) # fold_idx == None -> train 전체 호출.
        # self.total_size = len(data_handler.getX()['train'])
        data_handler.set_dataset('total')
        self.total_size = len(data_handler.getX()['total'])
        trainloader = DataLoader(data_handler, batch_size=self.batch)
        
        model = next(self.getModel()) 
        model_ae = autoencoder(num_class=self.classes, model=model).to(self.device)
        model_ae.train()
        optimizer = torch.optim.Adam(model_ae.parameters()) 
        
        epoch     = 1
        bad_cnt   = 0
        min_loss  = 99999
        min_count = 1
        self._gamma = 0
        self.ae_epoch = 150 if self.model_name == 'SAE_3D' else 1000
        self.ae_data_num = len(data_handler.getX()['total'])

        while epoch < self.ae_epoch and bad_cnt < 30:
            writer_remove = True if epoch == 1 else False
            writer = self.initWriter(fold_idx=None, mode='preTrain', writer_remove=writer_remove)
            print('-'*46)
            epoch_loss = 0
            step_len   = len(trainloader)
            for step, (train_X, (disease, patient)) in enumerate(trainloader):  
                disease = list(disease)
                patient = list(patient.data.cpu().numpy())
                outputs = model_ae(train_X[0].unsqueeze_(1))

                # [Forward PP] 2. Compute MSE Loss
                loss = torch.nn.functional.mse_loss(outputs, train_X[0]).to(self.device)
                step_loss = loss.item()
                epoch_loss += step_loss*len(disease)

                # [Back PP] 3. Do Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"\rstep[{step+1}] MSEloss : {step_loss} - data[{patient}]", end=' ')
                del train_X, disease, patient

            print()
            epoch_loss_mean = epoch_loss/self.total_size
            self.printStatus(epoch, step, step_len, epoch_loss_mean,f"pre-train")
            if min_loss > epoch_loss_mean:
                print("Best AE model has been updated")
                min_loss = epoch_loss_mean
                best_epoch = epoch
                if self.model_name == 'SAE_3D':
                    best_model_wght = model_ae.encoder_out().state_dict()
                else:
                    best_model_wght = model_ae.encoder.state_dict()
                total_model_wght = model_ae.state_dict()
                bad_cnt = 0
                min_count += 1
                if not min_count%1:
                    self.saveModel('autoencoder', -1, best_epoch, best_model_wght, min_loss)
                    self.saveModel('autoencoder_total', -1, best_epoch, total_model_wght, min_loss)
                    print(f'model saved. min_loss : {min_loss}')
            else:
                bad_cnt += 1

            epoch += 1
            
            self.saveResult(epoch, 'train', None, epoch_loss_mean, metric='ae_pretrain')
            del epoch_loss, step_loss

            writer.add_scalars('MSELoss',{'AE_pretrain':epoch_loss_mean}, epoch)
            writer.add_scalars('MinMSELoss',{'AE_pretrain':min_loss}, epoch)

            if min_loss < 1e-3:
                break
            
        self.saveModel('autoencoder', -1, best_epoch, best_model_wght, min_loss)
        self.saveModel('autoencoder_total', -1, best_epoch, total_model_wght, min_loss)
        writer.close()
        del trainloader

    def test_ae_pre_train(self, data_handler):
        self.lr = self.pre_trained_lr
        self.ae_data_num = 483
        # self.optimizer_name = self.args.transfer_learning_optimizer

        # data_handler.set_dataset('train', data_num= 130 if self.ae_data_num == 500 else None) # fold_idx == None -> train 전체 호출.
        data_handler.setSkipList()
        data_handler.loadLabel_setData()

        data_handler.set_dataset('total')
        self.total_size = len(data_handler.getX()['total'])
        
        trainloader = DataLoader(data_handler, batch_size=self.pre_trained_batch)

        nii_mask_dir =  f"./data/Nifti/In/Transformed/OCTA_SRL_256_V2"
        nii_output_dir = f"./data/Nifti/In/Transformed/AECheck/Reconstructed/Data{self.ae_data_num}"
        os.makedirs(nii_output_dir, exist_ok=True)

        model_ae = next(self.loadModel("autoencoder_total"))
        model_ae.eval()
        
        print(f"{self.ae_data_num} data have been loaded.")
        patients_list = [id_ for id_ in data_handler.get_current_data()]
        # MinMax_dict = totalloader.dataset.MinMax
        
        for idx, ( test_X, _ ) in enumerate(trainloader): 
            patient = patients_list[idx]
            if patient in [10001, 10005, 10167, 10293, 10301, 10355, 10410, 10480]:
                # Get reconstructed result
                nii_input = test_X[0].reshape(192,-1,192).data.cpu().numpy()
                nii_output = model_ae(test_X[0].unsqueeze_(1)).data.cpu().numpy() # prediction            
                recon = nii_output[0][0].reshape(192, -1, 192) # get 3 dim data
                recon = recon.astype(np.float32)
                
                # Masking 
                mask_nii = nib.load(os.path.join(nii_mask_dir, f"{patient}.nii.gz")) # load masking volume
                m_aff, m_head = mask_nii.affine, mask_nii.header
                mask_arr = np.asarray(mask_nii.dataobj)
                mask_srl_arr = np.zeros(np.shape(mask_arr))
                mask_srl_arr[mask_arr>0] = 1
                
                recon = np.multiply(recon, mask_srl_arr)
                # recon = np.uint8((recon-recon.min())/(recon.max()-recon.min())*255)
                # masked_recon = np.multiply(recon, mask_srl_arr)

                img = nib.Nifti1Image(recon, affine=m_aff, header=m_head)
                nii_dir_dst_recon = os.path.join(nii_output_dir, f"{patient}_SRL_RECON.nii.gz")
                nib.save(img, nii_dir_dst_recon)
                
                print(f"{patient} saved.")
        print()

    def initWriter(self, fold_idx, mode, cnt=None, writer_remove = False):
        assert mode is not None, 'Set the wrtier mode. f"{#}fold" or "retrain".'
        tb = self.tensorboard_dir 
        if fold_idx is not None:
            writer_dir = os.path.join(f"{tb}/{mode}/fold{fold_idx}/batch{self.batch}/{self.optimizer_name}/lr{self.lr}/gm{self._gamma}/{cnt}")
        else:
            writer_dir = os.path.join(f"{tb}/{mode}/batch{self.batch}/{self.optimizer_name}/lr{self.lr}/gm{self._gamma}/{cnt}")

        if (self.pre_writer is not None) and writer_remove:
            os.system(f'rm -rf {self.pre_writer}')
        
        writer = SummaryWriter(writer_dir, comment=f"{self.loss_name}")
        self.pre_writer = writer_dir
        return writer

    def doActivation(self, prediction):
        if   self.loss_name == 'bce': hypothesis = nn.Sigmoid()(prediction)
        elif self.loss_name == 'mse': hypothesis = nn.Softmax(dim=1)(prediction)
        elif self.loss_name == 'nll': hypothesis = nn.LogSoftmax(dim=1)(prediction)
        elif self.loss_name == 'fcl': hypothesis = prediction
        elif self.loss_name == 'ce': hypothesis = prediction
        else : raise ValueError("Choose correct Activation Function")
        yield hypothesis

    def getLoss(self, w=None):
        if   self.loss_name=='ce'  : loss = nn.CrossEntropyLoss() # same as nn.LogSoftMax + nn.NLLLoss
        elif self.loss_name=='fcl' : loss = FocalLoss()
        elif self.loss_name=='nll' : loss = nn.NLLLoss(weight=w) # need nn.LogSoftMax
        elif self.loss_name=='bce' : loss = nn.BCELoss() # need nn.Sigmoid 
        elif self.loss_name=='mse' : loss = nn.MSELoss() # need Softmax + Argmax
        yield loss

    def getOptimizer(self, p, lr=None):  
        if   self.optimizer_name == "sgd"     : optimizer = SGD(params=p, lr=self.lr,weight_decay=self.wd)
        elif self.optimizer_name == "asgd"    : optimizer = ASGD(params=p, lr=self.lr, weight_decay=self.wd)
        # elif self.optimizer_name == "asgd"    : optimizer = ASGD(filter(lambda x: x.requires_grad, p),                                                      )
        elif self.optimizer_name == "rmsp"    : optimizer = RMSprop(params=p, lr=self.lr,weight_decay=self.wd)
        elif self.optimizer_name == "adam"    : optimizer = Adam(params=p, lr=self.lr,weight_decay=self.wd)
        # elif self.optimizer_name == "adamp"   : optimizer = AdamP(params=p, lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        elif self.optimizer_name == "adamw"   : optimizer = AdamW(params=p, lr=self.lr, weight_decay=self.wd)
        elif self.optimizer_name == "adagrad" : optimizer = Adagrad(params=p, lr=self.lr,weight_decay=self.wd)
        elif self.optimizer_name == "adadelta": optimizer = Adadelta(params=p, lr=self.lr,weight_decay=self.wd) 
        else : optimizer = None
        yield optimizer

    def getModel(self):
        if '3' in self.dimension:
            if   self.model_name == "VGG16_3D"  : model = VGG16_3D(self.classes)
            elif self.model_name == "CV5FC2_3D" : model = CV5FC2_3D(self.classes)
            elif self.model_name == "CV3FC2_3D" : model = CV3FC2_3D(self.classes)
            elif self.model_name == "Incept_3D" : model = Inception3_3D(num_classes=self.classes)
            elif "eff" in self.model_name.lower() : model = EfficientNet3D.from_name("efficientnet-b4", override_params={'num_classes': 2}, in_channels=1)
            elif "res" in self.model_name.lower() : model = generate_model(model_depth=self.args.res_depth, n_classes=self.classes)
            # elif 'vit' in self.model_name.lower() : model = VIT_3D(self.classes, self.is_transfered)
            else : raise ValueError("Choose correct model")
        else:
            if 'res' in self.model_name.lower():
                model = ResNet_2D(self.classes, self.is_transfered, self.args.res_depth)
            elif 'vgg' in self.model_name.lower():
                vgg_depth = int(self.model_name.split('_')[1])
                model = VGG_2D(self.classes, self.is_transfered, depth=vgg_depth)
            elif 'google' in self.model_name.lower():
                model = GOOGLE_2D(self.classes, self.is_transfered)
            elif 'incept' in self.model_name.lower():
                model = INCEPT_V3_2D(self.classes, self.is_transfered)
            elif 'eff' in self.model_name.lower():
                model = EFFICIENT_2D(self.classes, self.is_transfered)
            elif 'vit' in self.model_name.lower():
                model = VIT_2D(self.classes, self.is_transfered)
        yield model.to(self.device)

    def loadModel(self, phase, fold_idx=None):
        '''phase : autoencoder/autoencoder_total/best/retrain'''
        import collections
        if phase in ['best', 'retrain'] :
            ae = 'ae_o' if self.is_transfered else 'ae_x'
            model_dir = os.path.join(self.check_dir, phase, ae)
            fold_dir = os.path.join(model_dir, "fold"+str(fold_idx))
            os.makedirs(fold_dir, exist_ok=True)
            name = f"model_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.pth"
            model_path = os.path.join(fold_dir, name)
        else:
            model_dir = os.path.join(self.check_dir, 'autoencoder')
            assert os.path.isdir(model_dir), f'{model_dir} is not exist.'
            bch = self.pre_trained_batch
            lr = self.pre_trained_lr
            
            if phase =='autoencoder_total':           
                name = f"total_b{bch}_{self.optimizer_name}_{self.loss_name}_{lr:.0E}_{self.ae_data_num}.pth"
            elif phase =='autoencoder':    
                name = f"model_b{bch}_{self.optimizer_name}_{self.loss_name}_{lr:.0E}_{self.ae_data_num}.pth" 
            model_path = os.path.join(model_dir, name)
            
        try: 
            # init_model
            print(f'model path : {model_path}')
            model = next(self.getModel())
            optimizer = next(self.getOptimizer(model.parameters()))
            
            # load model
            checkpoint = torch.load(model_path)
            self.best_epoch = checkpoint['epoch']
            if phase in ["autoencoder", "autoencoder_total"]:
                if phase == "autoencoder":
                    try: # this is not total model of encoder.
                        model.convolutions.load_state_dict(checkpoint['model_state_dict'])
                    except:
                        d = collections.OrderedDict()
                        for j in checkpoint['model_state_dict']:
                            if 'encoder' in j:
                                d[j.replace('encoder.','')] = checkpoint['model_state_dict'][j]
                        model.convolutions.load_state_dict(d)
                else:
                    if 'SAE' not in self.model_name :
                        model = autoencoder(num_class=self.classes, model=model).to(self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"AE pre-trained model has been loaded.")
                print(f"AE pre-trained model has been loaded.")
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                if phase == 'best' and self.is_transfered and '3' in self.dimension: # freezing when ae_transfer learning
                    model = freeze(num_class=self.classes, model=model).to(self.device)
                logging.info(f"Best model[{fold_idx}] has been loaded.")
                print(f"Best model[{fold_idx}] has been loaded.")

        except: 
            logging.info("!!! Loading model has been failed !!!")
            print("!!! Loading model has been failed !!!")

        yield model

    def saveModel(self, phase, fold_idx, epoch, model_state_dict, loss):
        '''phase : autoencoder/autoencoder_total/best/retrain'''

        if phase in ['best', 'retrain'] :
            ae = 'ae_o' if self.is_transfered else 'ae_x'
            model_dir = os.path.join(self.check_dir, phase, ae)
            fold = f"fold{fold_idx}" if fold_idx is not None else "single_train" 
            fold_dir = os.path.join(model_dir, f"{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            name = f"model_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.pth"
            model_path = os.path.join(fold_dir, name)
        else:
            model_dir = os.path.join(self.check_dir, 'autoencoder')
            os.makedirs(model_dir, exist_ok=True) 
            if phase =='autoencoder_total':           
                name = f"total_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}_{self.ae_data_num}.pth"
            elif phase =='autoencoder':    
                name = f"model_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}_{self.ae_data_num}.pth"
            model_path = os.path.join(model_dir, name)
        
        if os.path.isfile(model_path) : 
            os.system(f"rm {model_path}")
        print(f'model saved : {model_path}')
        torch.save({
                'epoch':epoch,
                'model_state_dict':model_state_dict,
                'loss':loss,
        }, model_path)

    def saveResult(self, epoch, status, fold_idx, loss_epoch_mean, metric, cfmx=None):
        result_dir = os.path.join(self.result_dir, status)
        if self.ae_pre_train and metric=='ae_pretrain':
            ae_result_dir  = f"{self.result_dir}/pretrain/ae_o"
            os.makedirs(ae_result_dir, exist_ok=True)
            ae_result_path = f"pretrain_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.txt"
            ae_result_name = os.path.join(ae_result_dir, ae_result_path)
            with open(ae_result_name, "a") as f:
                f.write(f"Epoch[{epoch}] MSELoss:{loss_epoch_mean:5g}\n")
        else:
            ae   = "ae_o" if self.is_transfered else "ae_x"
            if fold_idx==0 and not None: 
                raise ValueError("fold_idx should be larger than 0.")
            fold = f"fold{fold_idx}" if fold_idx is not None else "single_train"
            ae_result_dir = os.path.join(result_dir, f"{ae}")
            fold_result_dir = os.path.join(ae_result_dir, f"{fold}")

            os.makedirs(fold_result_dir, exist_ok=True)

            if self.classes == 2 :
                fold_result_path = f"{status}_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.txt"
                fold_result_name = os.path.join(fold_result_dir, fold_result_path)

                tp,tn,fp,fn = metric['TP'] ,metric['TN'] ,metric['FP'] ,metric['FN']
                ba,se,sp,f1 = metric['BA'] ,metric['SE'] ,metric['SP'] ,metric['F1']
                acc,pcs,rcl = metric['ACC'],metric['PCS'],metric['RCL']

                with open(fold_result_name, "w") as f:
                    f.write(f"Epoch[{epoch}]-{status}-Loss:{loss_epoch_mean:5g}\n")
                    if status=='retrain':
                        f.write(f"TP/TN/FP/FN - {tp}/{tn}/{fp}/{fn} Best-E/F:{self.best_epoch}/{fold_idx}\n")
                    else:
                        f.write(f"TP/TN/FP/FN - {tp}/{tn}/{fp}/{fn}\n")
                    if (tp+fn) != 0 and (tn+fp) !=0 :
                        f.write(f"{'SE':3}:{se:.5f}, {'SP':3}:{sp:.5f}\n{'BA':3}:{ba:.5f}, {'ACC':3}:{acc:.6f}\n")
                        f.write(f"{'PCS':3}:{pcs:.5f}, {'RCL':3}:{rcl:.5f}, {'F1':3}:{f1:.5f}\n")
                    f.write(f"\n")
            else:
                fold_result_path = f"{status}_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.png"
                fold_result_name = os.path.join(fold_result_dir, fold_result_path)
                fold_result_txt_path = f"{status}_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.txt"
                fold_result_txt_name = os.path.join(fold_result_dir, fold_result_txt_path)

                confusion_matrix_df = pd.DataFrame(cfmx, 
                                                   index = [i for i in self.current_disease], 
                                                   columns = [i for i in self.current_disease])

                plt.title(f'{status} Confusion Matrix', fontsize=20)
                sns.heatmap(confusion_matrix_df, annot=True, fmt='d')
                plt.savefig(f'{fold_result_name}')
                plt.close('all')
                print(f'{status} confusion matrix saved at : {fold_result_name}')

                score = metric
                with open(fold_result_txt_name, "w") as f:
                    f.write(f"Epoch[{epoch}]-{status}-Loss:{loss_epoch_mean:5g}\n")
                    if type(score) == tuple:
                        score = score[0]
                    for k in score:
                        if k!='accuracy' and k!='Loss':
                            f.write(f'{k:12} : ')
                        try:
                            for metric in score[k]:
                                if 'support' == metric:
                                    f.write(f'num-{np.round(score[k][metric], 6):3} ')
                                else:
                                    f.write(f'{metric}-{np.round(score[k][metric], 6):08} ')
                            f.write('\n')
                        except:
                            f.write(f'{k:12} : {np.round(score[k], 6)}')
                            f.write('\n')
                    f.write(f"\n")

    def printDataNum(self,fold_idx, dataset):
        train_set = list(dataset['train'])
        valid_set = list(dataset['valid'])
        dataset_size = {'train':len(train_set), 'valid':len(valid_set)}
        print("-"*18+f" fold -{fold_idx:2d} "+"-"*18)
        print(f"trainset-{dataset_size['train']:>3d} : NO/AB-{train_set.count(0)}/{train_set.count(1)}")
        print(f"validset-{dataset_size['valid']:>3d} : NO/AB-{valid_set.count(0)}/{valid_set.count(1)}")
        print('trainset-NO/AB rate :', np.round(train_set.count(0)/train_set.count(1), 2))
        print('validset-NO/AB rate :', np.round(valid_set.count(0)/valid_set.count(1), 2))
        return dataset_size

    def printStatus(self, epoch, step, step_len, loss, status):
        if status=="AE pre-training":
            print(f"[{epoch:2d}/{self.ae_epoch}]",end=' ')
        try:
            print(f"{'Epoch':5}[{epoch:2d}/{self.ae_epoch}]",end=' ')
        except:
            print(f"{'Epoch':5}[{epoch:2d}/{self.epoch}]",end=' ')
        print(f"{'Step':4}[{step+1:2d}/{step_len:2d}]",end=' ')
        print(f"{'LOSS':4}[{loss:>5f}]-[{status}]")
