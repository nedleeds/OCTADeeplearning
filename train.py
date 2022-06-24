import os
import random
import copy
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchsummary
from sklearn.model_selection import KFold, StratifiedKFold

from torch.nn import parameter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, dataloader, dataset
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop
from torch.utils.tensorboard import SummaryWriter, writer
from efficientnet_pytorch_3d import EfficientNet3D

from data import Data_Handler

from utils.resnet import generate_model
from utils.evaluate import checking
from utils.FocalLoss import FocalLoss
from utils.earlyStop import EarlyStopping
from utils.getOptimized import RandomSearch
from utils.INCEPT_V3_3D import Inception3_3D

from model import VGG16_2D, ResNet_2D, GOOGLE_2D, INCEPT_V3_2D, VGG_2D, EFFICIENT_2D, VIT_2D
from model import CV3FC2_3D, CV5FC2_3D,  VGG16_3D, Res50_3D, ResNet_3D, autoencoder, freeze

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
        self.preTrain       = True if args.ae_pre_train=='True' else False
        
        self.best_epoch     = 1
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.valid_subsampler = []
        self.classes = 2
    
        self.log_dir        = ''
        self.check_dir      = ''
        self.result_dir     = ''
        self.best_param_dir = ''
        self.tensorboard_dir= ''
        self.pre_writer = None
        self.tf_lrn_opt = args.transfer_learning_optimizer

    def __call__(self, data_handler):
        ae = 'ae_o' if self.is_transfered else 'ae_x'
        self.best_param_dir = os.path.join(self.best_param_dir, ae)
        self.log_dir        = data_handler.getOuputDir()['log']
        self.check_dir      = data_handler.getOuputDir()['checkpoint']
        self.result_dir     = data_handler.getOuputDir()['result']
        self.best_param_dir = os.path.join(data_handler.getOuputDir()['best_parameter'], ae)
        self.tensorboard_dir= os.path.join(data_handler.getOuputDir()['tensorboard'], ae)
        self.roc_plot_dir= os.path.join(data_handler.getOuputDir()['roc_plot'], ae)

        self.input_shape   = data_handler.getInputShape()
        self.disease_index = data_handler.sort_table(reverse=False)
        self.index_disease = data_handler.sort_table(reverse=True)
        self.label_table   = [self.disease_index, self.index_disease]

        self.fold_Best     = [0]*self.fold_num
        self.previous_BEST = [-np.inf for _ in range(self.fold_num)]
        self._best_parameter = {f'fold{idx}':{} for idx in range(1, self.fold_num+1)}    
        self.y_folds = []
        self.y_preds_folds = []

        self.pre_trained_batch = 1
        self.pre_trained_lr = self.args.ae_learning_rate#5e-2 #3e-3
        self.classes = len(data_handler.get_disease_keys())
        # self.modelSummary(dimension='3d', input_shape=(1,224,400,400))
        # checking model
        logging.info(f"Status-{list(set(self.index_disease.values()))}")

        if self.preTrain and ('3' in self.dimension):
            seed_everything(99) ## 5 for my AE
            ## using total set
            logging.info("Start AutoEncoder pre-processing.")
            self.AEpreTrain(data_handler)
            # self.testAEpreTrain(data_handler)
            return 
            # del totalset, totalloader
        
        # do stratified k-fold
        logging.info("Start stratified K-Fold Cross Validation.")
        self.total_Best_cm = [0]*self.fold_num

        if self.isMerge:
            self.classes = 2
            self.doKFold(data_handler)
            # self.retrain(data_handler) 
        else:
            if 'res_50' in self.model_name.lower():
                mp.spawn(self.multi_class(data_handler),
                         args=(world_size,),
                         nprocs=world_size,
                         join=True)
            else:
                self.multi_class(data_handler)
            # self.retrain(data_handler) 
        # return
        # self.doKFold(data_handler)
        # self.saveAVGResults(self.total_Best_cm) # save mean metric values from each folds.
        # self.saveTestMetric('valid', self.total_Best_cm)

        # from utils.getBestParam import getBestParam
        # self._best_parameter = getBestParam(self.args)

        # logging.info("Start Retrain process.")
        # self.retrain(data_handler) 

    def getBestFold(self):
        return np.argmax([self.total_Best_cm[x]['ACC'] for x in range(5)])

    def getBestParams(self):
        return self._best_parameter

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed()%2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def modelSummary(self, dimension, model=None, input_shape=None):
        assert dimension in ['2d', '3d'], "Set correct dimension."
        if model is not None : 
            pass
        else:
            model = next(self.getModel())
        if input_shape is not None:
            torchsummary.summary(model, input_shape)
        else:
            torchsummary.summary(model, self.input_shape)
            
    def check_normalization(self, train_X, disease, patient):
        disease = list(disease)[0]
        patient = list(patient.data.cpu().numpy())[0]
        if patient in [ 10035, 10057, 10114, 10219, 10220,10039, 10046, 10055, 10074, 10080,
                        10089,	10111, 10212, 10224, 10257, 
                        10285, 10288]:
            from PIL import Image

            # 1. og, 2. min_max, 3. meanfpr
            os.makedirs(enface_dir, exist_ok=True)

            nii_name_og = f'{patient}.nii.gz'
            nii_name_min_max = f'{patient}_{disease}_min_max.nii.gz'
            nii_name_prune_min_max = f'{patient}_{disease}_prune_min_max.nii.gz'

            img_name_og = f'{patient}_{disease}.png'
            img_name_min_max = f'{patient}_{disease}_min_max.png'
            img_name_prune_min_max = f'{patient}_{disease}_prune_min_max.png'

            nii_save_path_og = os.path.join(nii_dir, nii_name_og)
            nii_save_path_min_max = os.path.join(nii_dir, nii_name_min_max)
            nii_save_path_prune_min_max = os.path.join(nii_dir, nii_name_prune_min_max)

            #### Get Nifti - OG, Min/Max Normalized, Mean Normalized. ####
            arr_3d_mask = np.asarray(nib.load(os.path.join(mask_dir, f'{patient}.nii.gz')).dataobj)
            arr_3d_og = np.uint8(nib.load(os.path.join(og_dir, f'{patient}.nii.gz')).dataobj)
            arr_3d_og_srl = np.zeros(np.shape(arr_3d_og))
            arr_3d_og_srl[(0<arr_3d_mask) & (arr_3d_mask<5)] = arr_3d_og[(0<arr_3d_mask) & (arr_3d_mask<5)]
            # arr_3d_min_max = train_X[0].data.cpu().numpy()
            m = np.min(arr_3d_og_srl)
            M = np.max(arr_3d_og_srl)
            arr_3d_min_max = (arr_3d_og_srl - m) / (M-m)
            arr_3d_min_max = np.uint8(arr_3d_min_max.reshape((192,-1,192))*255)
            
            arr_3d_prune = np.zeros(np.shape(arr_3d_mask))
            lower = np.percentile(arr_3d_og_srl[np.where(arr_3d_og_srl > 0)], 0)
            upper = np.percentile(arr_3d_og_srl[np.where(arr_3d_og_srl > 0)], 99.99)
            arr_3d_prune[(lower <= arr_3d_og_srl) & (arr_3d_og_srl <= upper)] = arr_3d_og_srl[(lower <= arr_3d_og_srl) & (arr_3d_og_srl <= upper)]
            
            arr_3d_prune_min_max = np.zeros(np.shape(arr_3d_mask))
            m = np.min(arr_3d_prune)
            M = np.max(arr_3d_prune)
            arr_3d_prune_min_max = np.uint8((arr_3d_prune - m) / (M-m)*255)

            print(f'og - mean, max, min : {np.round(np.mean(arr_3d_og_srl[np.where(arr_3d_og_srl > 0)]),3), np.max(arr_3d_og[np.where(arr_3d_og_srl > 0)]), np.min(arr_3d_og[np.where(arr_3d_og_srl > 0)])}')
            print(f'min max - mean, max, min : {np.round(np.mean(arr_3d_min_max[np.where(arr_3d_og_srl > 0)]),3), np.max(arr_3d_min_max[np.where(arr_3d_og_srl > 0)]), np.min(arr_3d_min_max[np.where(arr_3d_og_srl > 0)])}')
            print(f'min max clip - mean, max, min : {np.round(np.mean(arr_3d_prune_min_max[np.where(arr_3d_og_srl > 0)]),3), np.max(arr_3d_prune_min_max[np.where(arr_3d_og_srl > 0)]), np.min(arr_3d_prune[np.where(arr_3d_og_srl > 0)])}')

            #### Convert numpy array to nibabel image & Save nibabel image ####
            ref = nib.load('./data/Nifti/In/Transformed/OCTA_SRL_256/10001.nii.gz')
            nii_img_og = nib.Nifti1Image(arr_3d_og_srl, ref.affine, ref.header)
            nii_img_min_max = nib.Nifti1Image(arr_3d_min_max, ref.affine, ref.header)
            nii_img_prune_min_max = nib.Nifti1Image(arr_3d_prune_min_max, ref.affine, ref.header)

            nib.save(nii_img_og, nii_save_path_og)
            nib.save(nii_img_min_max, nii_save_path_min_max)
            nib.save(nii_img_prune_min_max, nii_save_path_prune_min_max)

            #### Make En-face through MIP ####
            arr_2d_og = np.max(arr_3d_og_srl, axis=1)
            arr_2d_min_max = np.max(arr_3d_min_max, axis=1)
            arr_2d_prune_min_max = np.max(arr_3d_prune_min_max, axis=1)

            #### Convert numpy array to PIL image & Save PIL image ####
            pil_img_og = Image.fromarray(np.rot90(np.uint8(arr_2d_og)))
            pil_img_min_max = Image.fromarray(np.rot90(np.uint8(arr_2d_min_max)))
            pil_img_prune_min_max = Image.fromarray(np.rot90(np.uint8(arr_2d_prune_min_max)))
            
            #### Save 
            pil_img_og.save(os.path.join(enface_dir, img_name_og), "PNG")
            pil_img_min_max.save(os.path.join(enface_dir, img_name_min_max), "PNG")
            pil_img_prune_min_max.save(os.path.join(enface_dir, img_name_prune_min_max), "PNG")
            

            font = {'family' : 'normal',
                    'weight' : 'bold',
                    'size'   : 15}

            matplotlib.rc('font', **font)

            fig, axes = plt.subplots(3, 2, figsize=(15, 20), layout='constrained')

            hist_og, bin_edges_og = np.histogram(arr_3d_og_srl[np.where(arr_3d_og_srl > 0)])
            hist_min_max, bin_edges_min_max = np.histogram(arr_3d_min_max[np.where(arr_3d_og_srl > 0)])
            hist_prune_min_max, bin_edges_prune_min_max = np.histogram(arr_3d_prune_min_max[np.where(arr_3d_og_srl > 0)])


            axes[0,0].plot(bin_edges_og[0:-1], hist_og)
            i1 = axes[0,1].imshow(pil_img_og, cmap = 'gray')
            axes[1,0].plot(bin_edges_min_max[0:-1], hist_min_max)
            i2 = axes[1,1].imshow(pil_img_min_max, cmap = 'gray')
            axes[2,0].plot(bin_edges_prune_min_max[0:-1], hist_prune_min_max)
            i3 = axes[2,1].imshow(pil_img_prune_min_max, cmap = 'gray')

            axes[0,0].set_title(f"[{patient}] Input Data")
            axes[0,0].set_xlabel('grayscale value')
            axes[0,0].set_ylabel('pixel count')
            axes[0,1].set_title(f"{patient} En-Face")

            axes[1,0].set_title(f"[{patient}] Input + Min/Max Normalizing * 255")
            axes[1,0].set_xlabel('grayscale value')
            axes[1,0].set_ylabel('pixel count')
            axes[1,1].set_title(f"En-Face")

            axes[2,0].set_title(f"[{patient}] Input + Min/Max with clipping(upper:0.01%) * 255")
            axes[2,0].set_xlabel('grayscale value')
            axes[2,0].set_ylabel('pixel count')
            axes[2,1].set_title(f"En-Face")

            # plt.show()
            fig.colorbar(i1, ax = axes[0,1])
            fig.colorbar(i2, ax = axes[1,1])
            fig.colorbar(i3, ax = axes[2,1])
            plt.savefig(fname = os.path.join(hist_dir, img_name_og))
            plt.close()

            print(f"{patient} has been checked.")


    def testAEpreTrain(self, data_handler):
        
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

    def multi_class(self, data_handler):
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
            isGreat = False
            while not isGreat:
                # settings
                cnt += 1

                thsh = {'1':0.80,#87, 
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
                    # model = self.initFC(model, self.args, self.lr)
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
                    print('-'*46+f'\nEpoch {epoch}/{self.epoch} - cnt[{cnt}], thsh[{thsh[f"{fold_idx}"]}]')
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
                            # train_X = train_X[0]
                            # train_y = train_y[0].long()
                            # # print(f'step:{step}-{train_y}')
                            # # 매개변수 경사도를 0으로 설정
                            # optimizer.zero_grad()
                            # # 순전파
                            # # 학습 시에만 연산 기록을 추적
                            # with torch.set_grad_enabled(phase == 'train'):
                            #     if '3' in self.dimension : outputs = model(train_X.unsqueeze_(1))
                            #     else                     : outputs = model(train_X)

                            #     loss = nn.CrossEntropyLoss()(outputs, train_y)
                                
                            #     y_pred_softmax = torch.log_softmax(outputs, dim = 1)
                            #     _, prediction = torch.max(y_pred_softmax, dim = 1)    
                                

                            #     step_pd = prediction.data.cpu().numpy()
                            #     step_gt = train_y.data.cpu().numpy()

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

                                # 학습 단계인 경우 역전파 + 최적화
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                            
                            epoch_gt.extend(step_gt)
                            epoch_pd.extend(step_pd)
                            
                            # 통계
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
                        # 모델을 깊은 복사(deep copy)함
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
                                best_epoch_gt = epoch_gt
                                best_epoch_pd = epoch_pd
                                # self.roc_plot(fold_idx, epoch_gt, epoch_pd, phase)

                            if previous_loss >= epoch_loss_valid:
                                previous_loss = epoch_loss_valid
                            else:
                                worse_cnt+=1

                            writer.add_scalars('Loss',{'train':epoch_loss_train, 'valid':epoch_loss_valid}, epoch)
                            writer.add_scalars('ACC', {'train':metrics['train']['accuracy'],'valid':metrics['valid']['accuracy']}, epoch)
                    epoch += 1
                    writer.close()

                # thsh = {'1':0.87-cnt%100, '2': 0.87-cnt%100, '3':0.87-cnt%100, '4':0.84-cnt%100, '5':0.87-cnt%100} # seed 33
                if self.previous_BEST[fold_idx-1]>=thsh[f'{fold_idx}']:
                # if self.previous_BEST[fold_idx-1]>=0.8:
                    self.y_folds.extend(epoch_gt)
                    self.y_preds_folds.extend(epoch_pd)
                    self.saveModel('best', fold_idx, best_epoch, best_model_wts, best_loss)
                    # self.roc_plot(best_epoch_gt, best_epoch_pd, 'validation', fold_idx)
                    isGreat = True
                    break


    def doKFold(self, data_handler):
        check = checking(lss=self.loss_name, labels=self.label_table, isMerge=self.isMerge)
        for fold_idx in range(1, self.fold_num+1):
            seed_everything(34)
            data_handler.set_dataset('train', fold_idx=fold_idx)
            data_handler.set_dataset('valid', fold_idx=fold_idx)
            # self.disease_index = data_handler.sort_table(reverse=False)
            # self.index_disease = data_handler.sort_table(reverse=True)
            # self.label_table   = [self.disease_index, self.index_disease]
            # print(self.label_table)
            self.input_shape = data_handler.getInputShape()
            cnt = 0
            isGreat = False
            while not isGreat:
                # settings
                cnt += 1

                thsh = {'1':0.80,#90, 
                        '2':0.80,#90, 
                        '3':0.80,#90, 
                        '4':0.80,#84, 
                        '5':0.80}#91} # seed 33

                self.previous_BEST[fold_idx-1] = 0.
                self._gamma = 0.94

                # set the dataset
                dataset = { 'train' : iter(data_handler.gety()['train']) , 
                            'valid' : iter(data_handler.gety()['valid']) }
                dataset_sizes = self.printDataNum(fold_idx, dataset)

                if self.is_transfered and ('3' in self.dimension):
                    # loadModel <- load Best model from AE preTrained.
                    self.loss_name = 'nll'
                    self.optimizer_name = 'asgd'
                    model = next(self.loadModel("autoencoder"))
                    model = freeze(num_class=self.classes, model=model).to(self.device)
                    self.lr = self.args.learningrate # learning rate for transfer learning 
                    self.optimizer_name = self.tf_lrn_opt
                    self.loss_name = self.args.loss       
                    #sgd, asgd, rmsp, adam, adamw, adagrad, adadelta
                else : 
                    model = next(self.getModel())
                    # model = self.initFC(model, self.args, self.lr)
                    # freezing Convolution layers and set FC layer's required_grad True.
                    if '2' in self.dimension:
                        if 'vgg' in self.model_name:
                            for i,l in enumerate(model.vgg16.features):
                                if len(model.vgg16.features)-5<i<len(model.vgg16.features):
                                    l.requires_grad = True
                                else:
                                    l.requires_grad = False
                                    2
                optimizer = next(self.getOptimizer(model.parameters(), lr=self.lr))
                lss_class = next(self.getLoss())
                scheduler = StepLR(optimizer, step_size=10, gamma=self._gamma)
                best_model_wts = copy.deepcopy(model.state_dict())
                best_train = -9999
                worse_cnt = 0
                previous_loss = 9999.
                epoch = 1
                metrics = {'train':None, 'valid':None}
                earlyStop = EarlyStopping(key={'name':'Loss','mode':'min'}, tolerance=self.tolerance, patience=self.patience)
                best_epoch_gt = []
                best_epoch_pd = []
                if torch.cuda.device_count() > 1 and '50' in self.model_name:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
                while not earlyStop.step(metrics['valid']):
                    writer_remove = True if epoch == 1 else False
                    writer = self.initWriter(fold_idx, f'{self.fold_num}fold',cnt, writer_remove)
                    print('-'*46+f'\nEpoch {epoch}/{self.epoch} - cnt[{cnt}], thsh[{thsh[f"{fold_idx}"]}]')
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
                            train_X = train_X[0] if '2' in self.dimension else train_X[0].unsqueeze_(1).to(self.device)
                            train_y = train_y[0].long().to(self.device)
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

                                # 학습 단계인 경우 역전파 + 최적화
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                            
                            epoch_gt.extend(step_gt)
                            epoch_pd.extend(step_pd)
                            # self.printStatus(epoch, step, step_len, step_loss, 'TRAIN')
                            check.Step(step_gt, step_pd)
                            # check.showResult(step_gt, step_pd)
                            # 통계
                            step_loss = loss.item()
                            epoch_loss += step_loss*len(train_y)
                            del train_X, train_y, prediction
                        if phase == 'train':
                            scheduler.step()
                        epoch_loss_mean = epoch_loss/dataset_sizes[phase]
                        print(f'{phase} Loss : {round(epoch_loss_mean, 6)}', end=' ')
                        metric, cfmx = check.Epoch(epoch_gt, epoch_pd)
                        metrics[phase] = metric
                        metrics[phase]['Loss'] = epoch_loss_mean
                        # 모델을 깊은 복사(deep copy)함
                        self.saveResult(epoch, phase, fold_idx, epoch_loss_mean, metric)
                        
                        key_metric = 'F1'
                        if phase=='train' : 
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
                                best_epoch_gt = epoch_gt
                                best_epoch_pd = epoch_pd
                                # self.roc_plot(fold_idx, epoch_gt, epoch_pd, phase)

                            if previous_loss >= epoch_loss_valid:
                                previous_loss = epoch_loss_valid
                            else:
                                worse_cnt+=1

                            writer.add_scalars('Loss',{'train':epoch_loss_train, 'valid':epoch_loss_valid}, epoch)
                            writer.add_scalars('ACC', {'train':metrics['train']['ACC'],'valid':metrics['valid']['ACC']}, epoch)
                            writer.add_scalars('F1',  {'train':metrics['train']['F1'], 'valid':metrics['valid']['F1'] }, epoch)
                    epoch += 1
                    writer.close()

                # thsh = {'1':0.87-cnt%100, '2': 0.87-cnt%100, '3':0.87-cnt%100, '4':0.84-cnt%100, '5':0.87-cnt%100} # seed 33
                if self.previous_BEST[fold_idx-1]>=thsh[f'{fold_idx}']:
                # if self.previous_BEST[fold_idx-1]>=0.8:
                    self.y_folds.extend(epoch_gt)
                    self.y_preds_folds.extend(epoch_pd)
                    self.saveModel('best', fold_idx, best_epoch, best_model_wts, best_loss)
                    # self.roc_plot(best_epoch_gt, best_epoch_pd, 'validation', fold_idx)
                    isGreat = True
                    break
                    # if fold_idx == 1 :
                    #     return 
                    # return ###        

    def roc_plot(self, y, y_pred, phase, fold_idx):
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay, auc, roc_curve

        roc_dir = os.path.join(self.roc_plot_dir, phase)
        os.makedirs(roc_dir, exist_ok=True)                            
        name = f"roc_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}_{fold_idx}.png"
        roc_path = os.path.join(roc_dir, name)

        RocCurveDisplay.from_predictions(y, y_pred)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{phase} - Normal, Abnormal ROC curve.")
        plt.legend(loc="lower right")
        plt.savefig(roc_path)


        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # n_classes = 2

        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        # fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pred.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])        

        # # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(n_classes):
        #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # # Finally average it and compute AUC
        # mean_tpr /= n_classes

        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # # Plot all ROC curves
        # plt.figure()
        # plt.plot(
        #     fpr["micro"],
        #     tpr["micro"],
        #     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        #     color="deeppink",
        #     linestyle=":",
        #     linewidth=4,
        # )

        # plt.plot(
        #     fpr["macro"],
        #     tpr["macro"],
        #     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        #     color="navy",
        #     linestyle=":",
        #     linewidth=4,
        # )

        # colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(
        #         fpr[i],
        #         tpr[i],
        #         color=color,
        #         lw=lw,
        #         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        #     )

        # plt.plot([0, 1], [0, 1], "k--", lw=lw)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title(f"{phase} - Normal, Abnormal ROC curve.")
        # plt.legend(loc="lower right")
        # plt.savefig(roc_path)


    def retrain(self, data_handler):
        data_handler.set_dataset('train')
        dataset_sizes = len(data_handler.getX()['train'])
        learning_rate = self.lr
        if self.is_transfered : learning_rate = self.args.ae_learning_rate
        if self.isMerge:
            check = checking(lss=self.loss_name, labels=self.label_table, isMerge=self.isMerge)
        else:
            check = checking(lss=self.loss_name, labels=data_handler.getDiseaseLabel(), isMerge=self.isMerge)
            self.current_disease = sorted(list(data_handler.getDiseaseLabel().keys()))
        for fold_idx in range(1, self.fold_num+1):
            ch = True
            if ch: 
                print('-'*46)
                ch = False
            cnt = 0
            isGreat = False
            while not isGreat:
                cnt += 1
                previous_BEST = 0.   
                self._gamma = 0.94
                model = next(self.loadModel('best',fold_idx))
                optimizer = next(self.getOptimizer(model.parameters(), lr=learning_rate))
                lss_class = next(self.getLoss()) 
                scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
                best_model_wts = copy.deepcopy(model.state_dict())
                epoch = 1
                metrics = {'train':None}
                earlyStop = EarlyStopping(key={'name':'Loss','mode':'min'}, tolerance=self.tolerance, patience=self.patience)
                while not earlyStop.step(metrics['train']) and epoch < 501:
                    writer_remove = True if epoch == 1 else False
                    writer = self.initWriter(fold_idx, mode='retrain', writer_remove=writer_remove)
                    print('-'*46+'\nEpoch {}/{}'.format(epoch, self.epoch))
                    model.zero_grad()
                    for phase in ['train']:
                        epoch_gt, epoch_pd = [], []
                        model.train() 
                        epoch_loss = 0.0
                        trainloader = DataLoader(data_handler, batch_size=self.batch, shuffle=True)
                        step_len = len(trainloader)

                        for step, (train_X, train_y) in enumerate(trainloader):
                            train_X = train_X[0] if '2' in self.dimension else train_X[0].unsqueeze_(1)
                            train_y = train_y[0].long()
                            # 매개변수 경사도를 0으로 설정
                            optimizer.zero_grad()
                            # 순전파
                            # 학습 시에만 연산 기록을 추적
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(train_X)
                                if self.classes > 2:
                                    loss = nn.CrossEntropyLoss()(outputs, train_y)
                                    y_pred_softmax = torch.log_softmax(outputs, dim = 1)
                                    _, prediction = torch.max(y_pred_softmax, dim = 1)    
                                    step_pd = prediction.data.cpu().numpy()
                                else:
                                    if 'incept' in self.model_name.lower():
                                        prediction = next(self.doActivation(outputs[0]))
                                    else:
                                        prediction = next(self.doActivation(outputs))
                                    _, preds = torch.max(prediction, 1)
                                    loss = lss_class(prediction, train_y)
                                    step_pd = preds.data.cpu().numpy()

                                step_gt = train_y.data.cpu().numpy()

                                # 학습 단계인 경우 역전파 + 최적화
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                            
                            epoch_gt.extend(step_gt)
                            epoch_pd.extend(step_pd)
                            # self.printStatus(epoch, step, step_len, step_loss, 'TRAIN')
                            if self.classes <3:
                                check.Step(step_gt, step_pd)
                            # check.showResult(step_gt, step_pd)
                            # 통계
                            step_loss = loss.item()
                            epoch_loss += step_loss*len(train_y)
                            del train_X, train_y, prediction

                        if phase == 'train':
                            scheduler.step()

                        epoch_loss_mean = epoch_loss/dataset_sizes
                        self.printStatus(epoch, step, step_len, epoch_loss_mean,f"Best-E/F:{self.best_epoch}/{fold_idx}")
                        metric, cfmx = check.Epoch(epoch_gt, epoch_pd)
                        metrics[phase] = metric
                        metrics[phase]['Loss'] = epoch_loss_mean
                        self.saveResult(epoch, 'retrain', fold_idx, epoch_loss_mean, metric, cfmx)
                        # 모델을 깊은 복사(deep copy)함
                        
                        if self.classes == 2:
                            if phase=='train' : 
                                epoch_loss_train=epoch_loss_mean
                                if previous_BEST<= metrics[phase]['F1']:
                                    print(f'epoch[{epoch}] is best F1')
                                    self.saveResult(epoch, 'retrain', fold_idx, epoch_loss_train, metric)
                                    best_epoch = epoch
                                    best_loss = epoch_loss_train
                                    best_model_wts = copy.deepcopy(model.state_dict())
                                    previous_BEST = metrics[phase]['F1']

                                writer.add_scalars(f'{self.loss_name}',{'train':epoch_loss_mean}, epoch)
                        else:
                            thsh_key = 'f1'
                            if 'acc' in thsh_key:
                                metric_for_thresh = metric[thsh_key]
                            elif 'f1' in thsh_key:
                                metric_for_thresh = metric['macro avg']['f1-score']
                            # 모델을 깊은 복사(deep copy)함
                            
                            if phase=='train' : 
                                epoch_loss_train=epoch_loss_mean
                                if previous_BEST <= metric_for_thresh:
                                    previous_BEST = metric_for_thresh
                                    best_epoch = epoch
                                    best_loss = epoch_loss_train
                                    best_model_wts = copy.deepcopy(model.state_dict())

                    epoch += 1
                writer.close()
                if previous_BEST>0.85:
                    self.saveModel('retrain', fold_idx, best_epoch, best_model_wts, best_loss)
                    isGreat = True
                    break
            # if fold_idx == 1:
            #     return
            
   
    def AEpreTrain(self, data_handler):
        # args = self.args
        # args.disease = 'NORMAL AMD DR CNV CSC RVO OTHERS'
        # data_handler = Data_Handler(args)
        # data_handler(pre_train=True)
        self.lr = self.pre_trained_lr
        self.batch = self.pre_trained_batch
        # lr = 0.001(default) -> 0.0003 -> 0.003 -> 0.01 -> +scheduler
        # data_handler.set_dataset('train', data_num= 130 if self.ae_data_num == 500 else None) # fold_idx == None -> train 전체 호출.
        # self.total_size = len(data_handler.getX()['train'])
        data_handler.set_dataset('total')
        self.total_size = len(data_handler.getX()['total'])
        trainloader = DataLoader(data_handler, batch_size=self.batch)
        model = next(self.getModel()) 
        model_ae = model

        # 22.02.15.16:25 -
        if self.model_name != 'SAE_3D':
            model_ae  = autoencoder(num_class=self.classes, model=model).to(self.device)
        model_ae.train()
        optimizer = torch.optim.Adam(model_ae.parameters()) 
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.6)

        epoch     = 1
        bad_cnt   = 0
        min_loss  = 99999
        min_count = 1
        self._gamma = 0
        self.ae_epoch = 150 if self.model_name == 'SAE_3D' else 1000
        self.ae_data_num = len(data_handler.getX()['total'])

        ''' Noramalization Check
        for step, (train_X, (disease, patient)) in enumerate(trainloader):  
            self.check_normalization(train_X, disease, patient)
        return 
        '''

        while epoch < self.ae_epoch and bad_cnt < 30:
            writer_remove = True if epoch == 1 else False
            writer = self.initWriter(fold_idx=None, mode='preTrain', writer_remove=writer_remove)
        # while epoch < self.ae_epoch :
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
                # scheduler.step()

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

    def initFC(self, model, args, lr):
        # for initializing FC Layer with learned model.
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        model_name     = args.model
        optimizer_name = args.optimizer.lower()
        loss_name      = args.loss.lower()
        batch          = args.batch
        dimension      = args.dimension

        init_check_dir = f"./checkpoint/initial/{model_name}/{dimension}"
        init_check_name = f"model_b{batch}_{optimizer_name}_{loss_name}_{lr:.0E}.pth"
        init_check_path = os.path.join(init_check_dir, init_check_name)
        init_check = torch.load(init_check_path)

        init_weight, init_bias = dict(), dict()
        cnt = 0 
        for k,v in init_check['model_state_dict'].items():
            if 'fc' in k:
                if 'weight' in k : init_weight[cnt]=v.data.cpu().numpy()
                else : 
                    init_bias[cnt]=v.data.cpu().numpy()
                    cnt+=1
        
        base_weight, base_bias = dict(), dict()
        cnt = 0
        for i in model.fc:
            if not isinstance(i, nn.Dropout):
                base_weight[cnt] = i.weight.data.cpu().numpy()
                base_bias[cnt]   = i.bias.data.cpu().numpy()
                cnt+=1

        params_weight, params_bias = dict(), dict()
        for i in range(len(init_weight)):
            params_weight[f'init{i}'] = init_weight[i].reshape(-1)
            params_weight[f'base{i}'] = base_weight[i].reshape(-1)
            params_bias[f'init{i}'] = init_bias[i].reshape(-1)
            params_bias[f'base{i}'] = base_bias[i].reshape(-1)
        
        f, axs = plt.subplots(4,1, figsize=(10,15))
        for i in range(4):
            axs[i].set_title(f"nn.Linear[{i}]'s Bias", size=10)
            sns.distplot(params_bias[f'base{i}'], ax=axs[i], label='pytorch')
            sns.distplot(params_bias[f'init{i}'], ax=axs[i], label='trained')
            axs[i].legend()
        plt.suptitle('Bias Histogram',fontsize=30)
        plt.savefig('biasCompare.png')

        f, axs = plt.subplots(4,1, figsize=(10,15))
        for i in range(4):
            axs[i].set_title(f"nn.Linear[{i}]'s Weight", size=10)
            sns.distplot(params_weight[f'base{i}'], ax=axs[i], label='pytorch')
            sns.distplot(params_weight[f'init{i}'], ax=axs[i], label='trained')
            axs[i].legend()
        plt.suptitle('Weight Histogram',fontsize=30)
        plt.savefig('weightCompare.png')
        print()

        #init_check  dml 
        return init_check['load_state_dict']

    # def init_optimizer(self, p):
    #     """Initialize the optimizer and use checkpoint weights if resume is True."""
    #     optimizer = getattr(torch.optim, self.optimizer_name)(
    #         filter(lambda x: x.requires_grad, p),
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay,
    #     )
    #     return optimizer

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
        
    def getValidSet(self, trainset, valid_idx):
        valid_set = copy.deepcopy(trainset)
        valid_set.X = [valid_set.X[idx] for idx in valid_idx]
        valid_set.y = [valid_set.y[idx] for idx in valid_idx]
        valid_set.label = dict(list(valid_set.label.items())[idx] for idx in valid_idx)
        valid_set.NIFTIPATH = list(valid_set.getNiftiPath())
        valid_set.num_disease = valid_set.countDiseases(valid_set.label)
        self.len_valid = len(valid_set)
        yield valid_set

    def getTrainSet(self, trainset, train_idx):
        train_set = copy.deepcopy(trainset)
        train_set.X = [train_set.X[idx] for idx in train_idx]
        train_set.y = [train_set.y[idx] for idx in train_idx]
        train_set.label = dict(list(train_set.label.items())[idx] for idx in train_idx)
        train_set.NIFTIPATH = list(train_set.getNiftiPath())
        train_set.num_disease = train_set.countDiseases(train_set.label)
        self.len_train = len(train_set)
        yield train_set

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
            elif self.model_name == "SAE_3D"    : model = SAE()
            elif self.model_name == "Incept_3D" : model = Inception3_3D(num_classes=self.classes)
            elif "eff" in self.model_name.lower() : model = EfficientNet3D.from_name("efficientnet-b4", override_params={'num_classes': 2}, in_channels=1)
            elif "res" in self.model_name.lower() : model = generate_model(model_depth=self.args.res_depth, n_classes=self.classes)# model = ResNet_3D(self.classes).to(self.device)
            # elif 'vit' in self.model_name.lower() : model = VIT_3D(self.classes, self.is_transfered)
            else : raise ValueError("Choose correct model")
        else:
            if 'res' in self.model_name.lower():
                model = ResNet_2D(self.classes, self.is_transfered, self.args.res_depth)
            elif 'vgg' in self.model_name.lower():
                vgg_depth = int(self.model_name.split('_')[1])
                model = VGG_2D(self.classes, self.is_transfered, depth=vgg_depth)
                # model = VGG16_2D(self.classes, self.is_transfered)
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
            #self.loss_name = 'nll'
            #self.optimizer_name='asgd'
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
        if self.preTrain and metric=='ae_pretrain':
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

    def saveAVGResults(self, total_Best_cm):
        f1,pcs,rcl,acc,ba,sp=[],[],[],[],[],[]
        ae = 'ae_o' if self.is_transfered else 'ae_x'
        result_dir = f"{self.result_dir}/valid/{ae}"
        os.makedirs(result_dir, exist_ok=True)
        file_name = os.path.join(result_dir, f"valid_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.txt")

        for i in range(len(total_Best_cm)):
            f1.append(total_Best_cm[i]['F1'])
            pcs.append(total_Best_cm[i]['PCS'])
            rcl.append(total_Best_cm[i]['RCL'])
            acc.append(total_Best_cm[i]['ACC'])
            ba.append(total_Best_cm[i]['BA'])
            sp.append(total_Best_cm[i]['SP'])

        f1_mean , pcs_mean, rcl_mean = np.mean(f1),  np.mean(pcs),  np.mean(rcl)
        acc_mean, ba_mean,  sp_mean  = np.mean(acc), np.mean(ba),   np.mean(sp)
        
        with open(file_name, 'w') as f:
            f.write(f"[{self.model_name}] - Dimension:{self.dimension}, Filter:{self.filter}, AE pre-train:{self.is_transfered}")
            f.write(f"{'SP':3}:{sp_mean:.5f}\n{'BA':3}:{ba_mean:.5f}, {'ACC':3}:{acc_mean:.6f}\n")
            f.write(f"{'PCS':3}:{pcs_mean:.5f}, {'RCL':3}:{rcl_mean:.5f}, {'F1':3}:{f1_mean:.5f}\n")
            f.write(f"\n")


    def saveTestMetric(self, status, folds_metric):
        '''
        parameters
            status : 'test'
            metric : metrics of all folds
        function
            mode in ['mean', 'best']
            path = getPath(status)
            metric = sumFoldResult(metric, mode)
            saveMetric(path, metric, mode)
        return
            None
        '''
        def getPath(status, mode=None):
            ae = 'ae_o' if self.is_transfered else 'ae_x'
            
            result_dir = os.path.join(self.result_dir, ae, mode)
            os.makedirs(result_dir, exist_ok=True)
            file_path = os.path.join(result_dir, f"{status}_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.txt")
            return file_path

        def sumFoldResult(metric, mode=None):
            f1_list, pcs_list, rcl_list = [], [], []
            acc_list, sp_list, se_list = [], [], []
            
            for i in range(fold_size):
                f1_list.append(metric[i]['F1'])
                pcs_list.append(metric[i]['PCS'])
                rcl_list.append(metric[i]['RCL'])
                acc_list.append(metric[i]['ACC'])
                sp_list.append(metric[i]['SP'])
                se_list.append(metric[i]['SE'])

            if mode == 'mean':
                f1, pcs, rcl = np.mean(f1_list), np.mean(pcs_list), np.mean(rcl_list)
                acc, sp , se = np.mean(acc_list), np.mean(sp_list), np.mean(se_list)
            else:
                best_idx = np.argmax(f1_list)
                f1, pcs, rcl = f1_list[best_idx],  pcs_list[best_idx], rcl_list[best_idx]
                acc, sp,  se = acc_list[best_idx], sp_list[best_idx],  se_list[best_idx]
                
            return f1, pcs, rcl, acc, sp, se

        def saveMetric(path, metric, mode=None):
            f1, pcs, rcl, acc, sp, se = metric
            with open(path, 'w') as f:
                if mode =='best':
                    f.write(f"[{self.model_name}]-BestFold[{best_idx}]-{self.dimension}-{self.filter}-AE:{self.is_transfered}\n")
                else:
                    f.write(f"[{self.model_name}]-Mean-{self.dimension}-{self.filter}-AE:{self.is_transfered}\n")
                f.write(f"{'SP':3}:{sp :.5f}, {'SE':3}:{se :.5f}, {'ACC':3}:{acc :.6f}\n")
                f.write(f"{'PCS':3}:{pcs :.5f}, {'RCL':3}:{rcl :.5f}, {'F1':3}:{f1 :.5f}\n")
                f.write(f"\n")

        best_idx, fold_size = 0, len(folds_metric)
        for m in ['mean', 'best']:
            metric_path = getPath(status, mode=m)
            metric=sumFoldResult(folds_metric, mode=m)
            saveMetric(metric_path, metric, mode=m)

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

###############
# right after train's step_gt to loss : for train and validate
                    # normedWeights = [1/list(step_gt).count(0) if list(step_gt).count(0) != 0 else 1, 1/list(step_gt).count(1) if list(step_gt).count(1) != 0 else 1]
                    # # normedWeights = [list(step_gt).count(1), list(step_gt).count(0)]
                    # normedWeights = torch.FloatTensor(normedWeights).to(self.device)
                    # loss_clss = next(self.getLoss(normedWeights))
