import logging
import os
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import random 
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
from torch.utils.data import DataLoader

from utils import medcam
from utils.vit import ViT
from utils.resnet import generate_model
from utils.evaluate import checking
from utils.FocalLoss import FocalLoss
from utils.getBestParam import getBestParam
from utils.INCEPT_V3_3D import Inception3_3D
from efficientnet_pytorch_3d import EfficientNet3D

from model import VGG16_2D, ResNet_2D, GOOGLE_2D, INCEPT_V3_2D, VGG_2D, EFFICIENT_2D, VIT_2D
from model import CV3FC2_3D, CV5FC2_3D,  VGG16_3D, Res50_3D, ResNet_3D, autoencoder, freeze

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class test():
    def __init__(self, args):
        self.args           = args
        self.model_name     = args.model
        self.optimizer_name = args.optimizer.lower()
        self.loss_name      = args.loss.lower()
        self.lr             = args.learningrate
        self.epoch          = args.epoch
        self.batch          = args.batch
        self.fold_num       = args.fold_num
        self.is_transfered  = args.tfl
        self.isMerge        = args.mergeDisease
        self.filter         = args.filter
        self.dimension      = args.dimension
        self.flt            = 'flt_o' if args.flatten else 'flt_x'
        self.ae             = 'ae_o' if self.args.ae else 'ae_x'
        
        self.previous_BEST  = [-99 for _ in range(5)]
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_epoch     = 0

        self.log_dir        = ''
        self.check_dir      = ''
        self.result_dir     = ''
        self.best_param_dir = ''
        self.tensorboard_dir= ''


    def __call__(self, data_handler):
        seed_everything(34)
        self.best_param_dir = os.path.join(self.best_param_dir, self.ae)
        self.log_dir        = data_handler.getOuputDir()['log']
        self.check_dir      = data_handler.getOuputDir()['checkpoint']
        self.result_dir     = data_handler.getOuputDir()['result']
        self.best_param_dir = os.path.join(data_handler.getOuputDir()['best_parameter'], self.ae)
        self.tensorboard_dir = os.path.join(data_handler.getOuputDir()['tensorboard'], self.ae)
        self.roc_plot_dir = os.path.join(data_handler.getOuputDir()['roc_plot'], self.ae)
        self.fold_min_loss = 0

        print(f'is Merge? : {self.args.mergeDisease}')
        self.classes = 2 if self.args.mergeDisease else len(data_handler.getDiseaseKeys())

        best = {'fold_idx':0, 'metric':None, 'cfmx':None, 'macro_f1':-99} 
        batch = 1 if self.args.medcam else self.batch

        for phase in ['valid', 'test']: #, 'test']:
            if phase == 'test':
                data_handler.set_dataset(phase) ###
                self.current_disease = sorted(list(data_handler.getDiseaseLabel().keys()))
                testloader  = DataLoader(data_handler, batch_size=batch)

            self.test_size     = len(data_handler.gety()['test'])
            self.input_shape   = data_handler.getInputShape()
            self.disease_index = data_handler.sort_table(reverse=False)
            self.index_disease = data_handler.sort_table(reverse=True)
            self.label_table   = [self.disease_index, self.index_disease]
            
            best_params = getBestParam(self.args)
            self.test_cm = [0]*self.fold_num

            # for fold_idx in range(1, self.fold_num+1):
            #     # self.batch = int(best_params[f'fold{fold_idx}']['Batch'])
            #     # self.batch = 4
            #     # self.lr = 3e-3
            #     self.Test(fold_idx, testloader) ###
            #     # return ####
            
            # self.saveTestMetric(status='test', folds_metric=self.test_cm)        

            total_gt = []
            total_pd = []
            for fold_idx in range(1, self.fold_num+1):
                self.gcam_dir = self.roc_plot_dir+(f"/attention_maps/{phase}/fold{fold_idx}" if fold_idx is not None else "single_train")
                if 'valid' in phase:
                    data_handler.set_dataset(phase, fold_idx)
                    testloader = DataLoader(data_handler, batch_size=batch)
                    self.current_disease = sorted(list(data_handler.getDiseaseLabel().keys()))

                    label = data_handler.getDiseaseLabel() if self.classes > 2 else self.label_table
                    check = checking(lss=self.loss_name, labels=label, isMerge=self.isMerge)        
                    gt, pd = self.get_gt_pd(fold_idx, testloader, phase, check)
                    total_gt.extend(gt)

                    if self.classes>2:
                        metric, cfmx = check.Epoch(gt, pd)
                        if metric['macro avg']['f1-score']>best['macro_f1']:
                            print(f"fold[{best[f'fold_idx']}] macro_f1 : [{np.round(best['macro_f1'], 6)}]",end=' -> ')
                            print(f"fold[{fold_idx}] macro_f1 : [{np.round(metric['macro avg']['f1-score'], 6)}]")
                            best['macro_f1'] = np.round(metric['macro avg']['f1-score'], 6)
                            best['fold_idx'] = fold_idx
                            best['metric'] = metric
                            best['cfmx'] = cfmx
                    else:
                        self.save_gt_pd(gt, pd, self.roc_plot_dir, phase+f'_f{fold_idx}')
                        metric, cfmx = check.Epoch(gt, pd)
                        if metric['F1']>best['macro_f1']:
                            print(f"fold[{best[f'fold_idx']}] f1 : [{np.round(best['macro_f1'], 6)}]",end=' -> ')
                            print(f"fold[{fold_idx}] f1 : [{np.round(metric['F1'], 6)}]")
                            best['macro_f1'] = np.round(metric['F1'], 6)
                            best['fold_idx'] = fold_idx
                            best['metric'] = metric
                            best['cfmx'] = cfmx
                        

                if 'test' in phase:
                    if fold_idx != best['fold_idx']:
                        pass
                    else:
                        label = data_handler.getDiseaseLabel() if self.classes > 2 else self.label_table
                        check = checking(lss=self.loss_name, labels=label, isMerge=self.isMerge)        
                        gt, pd = self.get_gt_pd(fold_idx, testloader, phase, check)
                        total_gt.extend(gt)
                        total_pd.extend(pd)
                        
                        if self.classes>2:
                            metric, cfmx = check.Epoch(gt, pd)
                            if metric['macro avg']['f1-score']>=best['macro_f1']:
                                print(f"fold[{best[f'fold_idx']}] macro_f1 : [{np.round(best['macro_f1'], 6)}]",end=' -> ')
                                print(f"fold[{fold_idx}] macro_f1 : [{np.round(metric['macro avg']['f1-score'], 6)}]")
                                best['macro_f1'] = np.round(metric['macro avg']['f1-score'], 6)
                                best['fold_idx'] = fold_idx
                                best['metric'] = metric
                                best['cfmx'] = cfmx
                        else:
                            self.save_gt_pd(gt, pd, self.roc_plot_dir, phase+f'_f{fold_idx}')
                            metric, cfmx = check.Epoch(gt, pd)
                            if metric['F1']>=best['macro_f1']:
                                print(f"fold[{best[f'fold_idx']}] f1 : [{np.round(best['macro_f1'], 6)}]",end=' -> ')
                                print(f"fold[{fold_idx}] f1 : [{np.round(metric['F1'], 6)}]")
                                best['macro_f1'] = np.round(metric['F1'], 6)
                                best['fold_idx'] = fold_idx
                                best['metric'] = metric
                                best['cfmx'] = cfmx

                model = next(self.getModel())
                del model
                torch.cuda.empty_cache()
                
            # average total gt & pd
            if self.classes>2:
                metric, cfmx = check.Epoch(total_gt, total_pd)
                loss_epoch_mean = self.fold_min_loss/self.fold_num
                self.saveTotalCFMX('total_valid', phase, loss_epoch_mean, metric, cfmx)
            self.save_gt_pd(total_gt, total_pd, self.roc_plot_dir, phase)

    def save_gt_pd(self, tot_gt, tot_pd, save_dir, phase):
        import pandas as pd
        os.makedirs(save_dir, exist_ok=True)
        save_name = f'{self.dimension}_{self.flt}_{self.ae}_{phase}.csv'
        save_path = os.path.join(save_dir, save_name)
        df = pd.DataFrame({'y': tot_gt, 'y_pred': tot_pd})

        df.to_csv(save_path, index=True, index_label='index', mode='w')
    
    def get_gt_pd(self, fold_idx, data_loader, phase, check=None):
        # check     = checking(lss=self.loss_name, labels=self.label_table, isMerge=self.isMerge)
        if 'test' in phase:
            # model = next(self.loadModel('retrain', fold_idx))
            model = next(self.loadModel('best', fold_idx))
        else:
            model = next(self.loadModel('best', fold_idx))
        attention_dir = f'{self.roc_plot_dir}/attention_maps/{phase}/fold{fold_idx}'
        names = [f'{attention_dir}/{s_id}_{d}' for s_id, d in data_loader.dataset.get_current_data().items()]
        shape = (192, 192) if '2' in self.dimension else (192, 256, 192)
        
        if self.args.medcam:
            model = medcam.inject(  model, 
                                    output_dir = self.gcam_dir, 
                                    save_maps = True,
                                    file_names = names,
                                    layer = 'auto')
        else:
            pass
        
        
        
        model.eval()   # 모델을 평가 모드로 설정
        
        lss_class = next(self.getLoss())
        metrics = {'test':None}            
        epoch_gt, epoch_pd = [], []

        epoch_loss = 0.0
        print(f"Fold[{fold_idx}] - ", end='')
        torch.cuda.empty_cache()
        for step, (test_X, test_y) in enumerate(data_loader):
            test_X = test_X[0] if '2' in self.dimension else test_X[0].unsqueeze_(1)
            test_y = test_y[0].long()                
            patient = names[step].split('/')[-1].split('_')[0]
            if '3' in self.dimension:
                nib_ref = nib.load(os.path.join(self.args.data_path, f'{patient}.nii.gz'))
                nib_header_info = { 'affine':nib_ref.affine,
                                    'header':nib_ref.header}
            else :
                nib_header_info = None
            
            # t_X = test_X.squeeze(0).squeeze(0)
            # t_X = t_X.data.cpu().numpy()
            # t_X = t_X.transpose(1,0,2)
            # t_X = nib.Nifti1Image(t_X, affine=nib_header_info['affine'], header=nib_header_info['header'])
            # nib.save(t_X, names[step] + "_og.nii.gz")

            
            with torch.no_grad():
                # model.eval()
                if self.args.medcam:
                    if '3' in self.dimension:
                        outputs = model(test_X, raw_input=test_X, nib_info=nib_header_info) # medcam
                    else:
                        outputs = model(test_X, raw_input=test_X)
                else:
                    outputs = model(test_X) # original
                if self.classes > 2:
                    loss = nn.CrossEntropyLoss()(outputs, test_y)
                    y_pred_softmax = torch.log_softmax(outputs, dim = 1)
                    _, prediction = torch.max(y_pred_softmax, dim = 1)    
                    step_pd = prediction.data.cpu().numpy()
                else:
                    prediction = next(self.doActivation(outputs))
                    _, preds = torch.max(prediction, -1)
                    loss = lss_class(prediction.reshape(-1,2), test_y)
                    
                    if len(test_y)>1:
                        step_pd = preds.data.cpu().numpy()
                    else:
                        step_pd = [preds.item()]
                    
                # if '2' in self.dimension:
                #     background_path = os.path.join(self.args.data_path, f"{patient}.png")
                #     overlay_path = f"{names[step]}.png"
                #     heatmap_path = f"{names[step]}_heatmap.png"
                #     background = cv2.imread(background_path)
                #     overlay = cv2.imread(overlay_path)
                #     overlay = cv2.resize(overlay, dsize=shape, interpolation=cv2.INTER_AREA)
                #     added_image = cv2.addWeighted(background, 0.8, overlay,0.2, 0)
                #     heatmap, img = self.overlay_heatmap(heatmap=overlay, image=background)
                #     cv2.imwrite(heatmap_path, heatmap)
                #     cv2.imwrite(overlay_path, img)
                        
                step_gt = test_y.data.cpu().numpy()                
                epoch_gt.extend(step_gt)
                epoch_pd.extend(step_pd)
                step_loss = loss.item()
                epoch_loss += step_loss*len(test_y)
                check.Step(step_gt, step_pd)
                # 통계
        if self.classes < 3:
            metric = check.Epoch(epoch_gt, epoch_pd)

        self.fold_min_loss += epoch_loss

        return epoch_gt, epoch_pd

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)        

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def Test(self, fold_idx, testloader):
        check = checking(lss=self.loss_name, labels=self.label_table, isMerge=self.isMerge)
        # model     = next(self.loadModel('retrain', fold_idx)) # for KFold, test
        model = next(self.loadModel('best', fold_idx)) 
        lss_class = next(self.getLoss())
        metrics = {'test':None}            
        epoch_gt, epoch_pd = [], []
        model.eval()   # 모델을 평가 모드로 설정
        epoch_loss = 0.0
        print(f"Fold[{fold_idx}] - ", end='')
        for step, (test_X, test_y) in enumerate(testloader):
            with torch.no_grad():
                test_X = test_X[0]
                test_y = test_y[0].long()
                if self.dimension == '3d' : outputs = model(test_X.unsqueeze_(1))
                else                      : outputs = model(test_X)
                prediction = next(self.doActivation(outputs))
                _, preds = torch.max(prediction, 1)
                loss = lss_class(prediction, test_y)

                step_pd = preds.data.cpu().numpy()
                step_gt = test_y.data.cpu().numpy()
                
                epoch_gt.extend(step_gt)
                epoch_pd.extend(step_pd)

                step_loss = loss.item()
                epoch_loss += step_loss*len(test_y)
                # self.printStatus(epoch, step, step_len, step_loss, 'TRAIN')
                check.Step(step_gt, step_pd)
                # check.showResult(step_gt, step_pd)
                # 통계

        epoch_loss_mean = epoch_loss/self.test_size
        metric = check.Epoch(epoch_gt, epoch_pd)
        self.test_cm[fold_idx-1] = metric
        self.saveResult(self.best_epoch, 'test', fold_idx, step, epoch_loss_mean, metric)

    def doActivation(self, prediction):
        if   self.loss_name == 'bce': hypothesis = nn.Sigmoid()(prediction)
        elif self.loss_name == 'mse': hypothesis = nn.Softmax(dim=1)(prediction)
        elif self.loss_name == 'nll':
            if prediction.shape==torch.Size([2]):
                hypothesis = nn.LogSoftmax()(prediction)
            else:
                hypothesis = nn.LogSoftmax(dim=1)(prediction)
        elif self.loss_name == 'fcl': hypothesis = prediction
        elif self.loss_name == 'ce' : hypothesis = prediction
        else : raise ValueError("Choose correct Activation Function")
        yield hypothesis

    def getLoss(self, w=None):
        if   self.loss_name=='ce'  : loss = nn.CrossEntropyLoss() # same as nn.LogSoftMax + nn.NLLLoss
        elif self.loss_name=='fcl' : loss = FocalLoss()
        elif self.loss_name=='nll' : loss = nn.NLLLoss(weight=w) # need nn.LogSoftMax
        elif self.loss_name=='bce' : loss = nn.BCELoss() # need nn.Sigmoid 
        elif self.loss_name=='mse' : loss = nn.MSELoss() # need Softmax + Argmax
        yield loss

    def getModel(self):
        if '3' in self.dimension:
            if   self.model_name == "VGG16_3D"  : model = VGG16_3D(self.classes)
            elif self.model_name == "CV5FC2_3D" : model = CV5FC2_3D(self.classes)
            elif self.model_name == "CV3FC2_3D" : model = CV3FC2_3D(self.classes)
            elif self.model_name == "SAE_3D"    : model = SAE()
            elif self.model_name == "Incept_3D" : model = Inception3_3D(num_classes=self.classes)
            elif "eff" in self.model_name.lower() : model = EfficientNet3D.from_name("efficientnet-b4", override_params={'num_classes': 2}, in_channels=1)
            elif "res" in self.model_name.lower() : model = generate_model(model_depth=self.args.res_depth, n_classes=self.classes)# model = ResNet_3D(self.classes).to(self.device)
            elif "vit" in self.model_name.lower() : model = ViT(in_channels=1, 
                                                                img_size=self.input_shape[1:], 
                                                                patch_size=(16, 16, 16),
                                                                # pos_embed='conv',
                                                                num_classes=2,
                                                                classification=True)
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

        try: 
            # init_model
            model = next(self.getModel())

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)     

            # load model
            checkpoint = torch.load(model_path)

            self.best_epoch = checkpoint['epoch']
            if phase in ["autoencoder", "autoencoder_total"]:
                if phase == "autoencoder":
                    try: # this is not total model of encoder.
                        model.convolutions.load_state_dict(checkpoint['model_state_dict'])
                        print('pre-trained conv weights have been loaded perfectly.')
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
                # if phase == 'best' and self.is_transfered and self.dimension == '3d': # freezing when ae_transfer learning
                #     model = freeze(num_class=self.classes, model=model).to(self.device)
                logging.info(f"Best model[{fold_idx}] has been loaded.")
                print(f"Best model[{fold_idx}] has been loaded.")

        except: 
            logging.info("!!! Loading model has been failed !!!")
            print("!!! Loading model has been failed !!!")
            print(f"model dir exist? : {os.path.isdir(model_dir)}")
            if os.path.isdir(model_dir) == False:
                print(f"no - {model_dir}")
            print(f"model path exist? : {os.path.isfile(model_path)}")
            if os.path.isfile(model_path) == False:
                print(f"no - {model_path}")
        yield model

    # def loadModel(self, phase, fold_idx=None):
    #     '''phase : autoencoder/autoencoder_total/best/retrain'''
    #     import collections
    #     if phase in ['best', 'retrain'] :
    #         ae = 'ae_o' if self.is_transfered else 'ae_x'
    #         model_dir = os.path.join(self.check_dir, phase, ae)
    #         # model_dir = './Data/output/checkpoint/fromServer/5fold/flt_o/best/ae_o'
    #         fold = f"fold{fold_idx}" if fold_idx is not None else "single_train" 
    #         fold_dir = os.path.join(model_dir, f"{fold}")
    #         print(f'model_dir exist : {os.path.isdir(fold_dir)}')
    #         if self.is_transfered == False:
    #             name = f"model_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.pth"
    #         else:
    #             name = f"model_b{self.batch}_{self.args.transfer_learning_optimizer}_{self.loss_name}_{self.lr:.0E}.pth"
    #         model_path = os.path.join(fold_dir, name)
    #         print(model_path)
    #         print(f'model_path exist : {os.path.isfile(model_path)}')
    #     else:
    #         model_dir = f'./checkpoint_ae/autoencoder'
    #         os.makedirs(model_dir, exist_ok=True)
    #         if phase == 'autoencoder_total':
    #             name = f"{self.model_name}_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}_11111_{self.ae_data_num}.pth"
    #         else:
    #             # name = f"model_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.pth"
    #             # name = f"model_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}_11111_{self.ae_data_num}.pth"
    #             name = f"{self.model_name}_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}_11111_{self.ae_data_num}.pth"
    #             # model_b4_asgd_nll_3E-03_11111
    #         model_path = os.path.join(model_dir, name)

    #     try: 
    #         # init_model
    #         model = next(self.getModel())
    #         # load model
    #         checkpoint = torch.load(model_path)
    #         print(f'model_path : {model_path}')
    #         print(f"Best Epoch/Loss : {checkpoint['epoch']}/{round(checkpoint['loss'],5)}")
    #         self.best_epoch = checkpoint['epoch']
    #         if phase in ["autoencoder", "autoencoder_total"]:
    #             if phase == "autoencoder":
    #                 try: # this is not total model of encoder.
    #                     model.convolutions.load_state_dict(checkpoint['model_state_dict'])
    #                 except:
    #                     d = collections.OrderedDict()
    #                     for j in checkpoint['model_state_dict']:
    #                         if 'encoder' in j:
    #                             d[j.replace('encoder.','')] = checkpoint['model_state_dict'][j]
    #                     model.convolutions.load_state_dict(d)
    #             else: 
    #                 model = autoencoder(num_class=self.classes, model=model).to(self.device)
    #                 model.load_state_dict(checkpoint['model_state_dict'])
    #             logging.info(f"AE pre-trained model has been loaded.")
    #             print(f"AE pre-trained model has been loaded.")
    #         else:
    #             model.load_state_dict(checkpoint['model_state_dict'])
    #             logging.info(f"Best model[{fold_idx}] has been loaded.")
    #             print(f"Best model[{fold_idx}] has been loaded.")

    #     except: 
    #         logging.info("!!! Loading model has been failed !!!")
    #         print("!!! Loading model has been failed !!!")

    #     yield model
    def get_figure_title(self):
        dimension = '2D VGG16' if '2' in self.dimension else '3D CNN'
        pre_train = 'pre-trained' if self.ae=='ae_o' else 'from scratch'
        return f'{dimension} {pre_train}'

    def get_f1_score(self, metric, cfmx):
        class_num = np.shape(cfmx)[0]
        disease = list(metric.keys())[:class_num]
        f1_score = [f"f1: {metric[d]['f1-score']*100:.2f}"for d in disease]
        f = []
        cnt = 0
        for arr in np.identity(n=class_num):
            f2 = []
            for a in arr:
                if a:
                    f2.append(f1_score[cnt])
                    cnt+=1
                else:
                    f2.append('')
            f+=f2
        # np.diag(f1_score).flatten()
        # return np.diag(f1_score)
        return f


    def saveTotalCFMX(self, epoch, status, loss_epoch_mean, metric, cfmx=None):
        os.makedirs(self.roc_plot_dir, exist_ok=True)
        save_png_name = f'{self.dimension}_{self.flt}_{self.ae}_{status}.png'
        save_txt_name = f'{self.dimension}_{self.flt}_{self.ae}_{status}.txt'
        
        save_png_path = os.path.join(self.roc_plot_dir, save_png_name)
        save_txt_path = os.path.join(self.roc_plot_dir, save_txt_name)
        confusion_matrix_df = pd.DataFrame( cfmx,
                                            index   = [i for i in self.current_disease], 
                                            columns = [i for i in self.current_disease])

        # make labels for cfms counts and percentage
        group_counts = ['{0:0.0f}'.format(value) for value in cfmx.flatten()]
        group_f1 = self.get_f1_score(metric, cfmx)
        labels = [f'{counts}\n{percentages}' for counts, percentages in zip(group_counts, group_f1)]
        labels = np.asarray(labels).reshape(np.shape(cfmx))

        status = 'Validation' if "valid" in status else 'Test'
        model_used = self.get_figure_title()
        sns.set(font_scale=1.2)
        plt.title(f'{status} result of {model_used}')
        # sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Greens')
        sns.heatmap(confusion_matrix_df, annot=labels, fmt='', cbar=False)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.savefig(f'{save_png_path}')
        plt.close('all')
        print(f'{status} confusion matrix saved at : {save_png_path}')

        score = metric
        with open(save_txt_path, "w") as f:
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


    def saveResult(self, epoch, status, fold_idx, num_step, loss_epoch, metric):
        result_dir = os.path.join(self.result_dir, status)
        if self.is_transfered: 
            test_result_dir  = os.path.join(result_dir, f"ae_o")
        else: 
            test_result_dir  = os.path.join(result_dir, f"ae_x")
            
        fold_result_dir=os.path.join(test_result_dir, f"kfold")
        os.makedirs(fold_result_dir, exist_ok=True)

        name = f"{status}_b{self.batch}_{self.optimizer_name}_{self.loss_name}_{self.lr:.0E}.txt"
        fold_result_path = os.path.join(fold_result_dir, name)

        tp,tn,fp,fn = metric['TP'] ,metric['TN'] ,metric['FP'] ,metric['FN']
        ba,se,sp,f1 = metric['BA'] ,metric['SE'] ,metric['SP'] ,metric['F1']
        acc,pcs,rcl = metric['ACC'],metric['PCS'],metric['RCL']

        with open(fold_result_path, "a") as f:
            f.write(f"Fold[{fold_idx}]/Epoch[{epoch}]-{status}-Loss:{loss_epoch/num_step:5g}\n")
            f.write(f"TP/TN/FP/FN - {tp}/{tn}/{fp}/{fn}\n")
            if (tp+fn) != 0 and (tn+fp) !=0 :
                f.write(f"{'SE':3}:{se:.5f}, {'SP':3}:{sp:.5f}\n{'BA':3}:{ba:.5f}, {'ACC':3}:{acc:.6f}\n")
                f.write(f"{'PCS':3}:{pcs:.5f}, {'RCL':3}:{rcl:.5f}, {'F1':3}:{f1:.5f}\n")
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
            result_dir = f"./result/{self.dimension}/{self.model_name}/{self.filter}/{status}"
            if self.is_transfered:
                result_dir = os.path.join(result_dir, 'ae_o')
            else: 
                result_dir = os.path.join(result_dir, 'ae_x')

            result_dir = os.path.join(result_dir, mode)
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

    def printStatus(self, epoch, step, step_len, loss, status):
        print(f"{'Epoch':5}[{epoch:4d}/{self.epoch}]",end=' ')
        print(f"{'Step':4}[{step+1:2d}/{step_len:2d}]"  ,end=' ')
        print(f"{'LOSS':4}[{loss:>5f}]", end='-')
        print(f"[{status}]")
