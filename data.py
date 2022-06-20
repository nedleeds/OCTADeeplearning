import enum
import os
import torch

import numpy as np
import pandas as pd
import nibabel as nib
import cv2

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torch.utils.data import Dataset


class Data_Handler(Dataset):
    def __init__(self, args):
        self.args = args
        self.__pre_train = (args.ae_pre_train == 'True')
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__input_data_path = args.data_path
        self.__input_label_path = args.label_path
        self.__test_rate = args.test_rate
        self.__dim = args.dimension
        self.__filter = args.filter
        self.__layer = args.layer
        self.__model = args.model
        self.__loss = args.loss
        self.__ae = args.ae
        self.__selected_disease = args.disease
        self.__isMerge = args.mergeDisease
        self.__fold_num = args.fold_num
        self.__seed = 304
        self.__label_pdFrame = {}
        self.__data_dict = {}
        self.__diseases_keys = {}
        self.__diseases = {'train':{}, 'test':{}, 'total':{}}
        self.__total_path = {'train':'', 'test':'', 'total':{}}
        self.__current_data_dict = {}
        self.__phase = ''
        self.__X = {'train':[], 'valid':[], 'test':[], 'total':[]}
        self.__y = {'train':[], 'valid':[], 'test':[], 'total':[]}
        self.__input_shape = None
        self.__3d_minMax_values = {}
        self.__disease_label = {}
        self.__disease_dir = ''
        self.__flatten = args.flatten
        self.__patients = []
        self.__ae_data_num = args.ae_data_num
        self.is_patch = args.patch_mode
        self.data_path = args.data_path

    def __call__(self):
        self.setSkipList()
        self.checkArguments()
        self.loadLabel_setData()
        self.splitTrainTest()
        self.doKFold()
        self.setOutputDir()
        print()

    def checkArguments(self):
        '''
        check path and arguments. (dim=2d/3d, flter=OG/Curvelet, layer=SRL/)
        '''
        dim = self.__dim
        flter = self.__filter
        layer = self.__layer
        data_path = self.__input_data_path
        model = self.__model

        if dim=='2d':
            assert 'dataset' in data_path, f"Set correct data_path({data_path}) for dimension({dim})."
            if flter == 'OG':
                assert 'OG' in data_path, f"Set correct data_path({data_path}) for filter({flter})."
            else:
                assert 'Curvelet' in data_path, f"Set correct data_path({data_path}) for filter({flter})."

            assert '2d' in model.lower(), f"Set corret model({model}) for {dim}."

        else:
            assert 'Nifti' in data_path, f"Set correct data_path({data_path}) for dimension({dim})."
            if flter == 'OG':
                if layer == 'SRL':
                    assert 'SRL' in data_path.upper(), f"Set correct data_path({data_path}) for filter({flter})."
                elif layer == 'DRL':
                    assert 'DRL' in data_path, f"Set correct data_path({data_path}) for filter({flter})."
                elif layer =='Total':
                    assert 'OCTA'in data_path, f"Set correct data_path({data_path}) for filter({flter})."
                else:
                    raise ValueError(f"Set correct layer({layer}) for filter({flter}), dim({dim})")

            else:
                assert 'Curvelet' in data_path, f"Set correct data_path({data_path}) for filter({flter})."
                # curvelet --> need to make SRL, DRL, total

            assert '3d' in model.lower(), f"Set corret model({model}) for {dim}."

    def setSkipList(self):
        skip = []
        # skip = [10219, 10220]
        skip = [10035, 10057, 10114, 10219, 10220]
        if self.__pre_train:
            skip.extend([10039, 10046, 10055, 10074, 10080,
        	             10089,	10111, 10212, 10224, 10257, 
                         10285, 10288])
        # 10035, 10039, 10057, 10074, 10080, 
        # 10089, 10114, 10219, 10220, 10224, 
        # 10257, 10285, 10288

        self.__skipList = skip

    def loadLabel_setData(self):
        label = self.loadLabel()
        self.__label_pdFrame = label

        for patient, disease in iter(zip(label['ID'], label['Disease'])):
            if int(self.__ae_data_num) == 500 and self.__pre_train and (patient not in self.__skipList):
                self.setDataDict(patient, disease)
                self.setDiseaseKeys()
            else:
                if (disease in self.__selected_disease) and (patient not in self.__skipList):
                    self.setDataDict(patient, disease)
                    self.setDiseaseKeys()
                else: 
                    pass

    def splitTrainTest(self):
        print(f'{"[ Train/Test Split ]":=^60}')
        if self.__test_rate == 0.1 :
            splits=10
        elif self.__test_rate ==0.15:
            splits=7
        elif self.__test_rate == 0.2:
            splits=6
        elif self.__test_rate == 0.3:
            splits=4
        elif self.__test_rate == 0:
            splits=30
        else:
            raise ValueError("Test Rate should be in [0, 0.1, 0.15, 0.2, 0.3]")

        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.__seed)
        patients = list(self.getDataDict().keys())
        diseases = list(self.getDataDict().values())
        for train_indices, test_indices in skf.split(patients, diseases):
            self.countDisease(diseases, train_indices, test_indices)
            self.saveSplitInfo(patients, train_indices, test_indices)
            print()
            break

    def doKFold(self):
        '''
        for the case of imbalanced data set classes,
        use 'Stratified KFold classification'.
        '''
        train_path = self.getTotalPath()['train']
        train_pdFrame = pd.read_csv(train_path)
        train_patients = list(train_pdFrame['patient'])
        train_disease = list(train_pdFrame['disease'])

        skf = StratifiedKFold(n_splits=self.__fold_num, shuffle=True, random_state=self.__seed)
        fold_idx = 1
        for train_indices, validation_indices in skf.split(train_patients, train_disease):
            print(f'{f"[ fold {fold_idx} ]":=^60}')
            self.countDisease(train_disease, train_indices, validation_indices)
            self.saveSplitInfo(train_patients, train_indices, validation_indices, fold_idx)
            fold_idx+=1

        self.checkFolds()
        print()

    def readCSV(self, phase, fold_idx=None):
        if fold_idx==None:
            if phase == 'total':
                self.setTotalPath('total', './Data/input/total.csv')
            path = self.getTotalPath()[phase]
        else:
            path = self.getTotalPath()[f'fold{fold_idx}'][phase]
        
        pdFrame = pd.read_csv(path)
        data_dict = list(pdFrame['patient'])
        return data_dict

    def checkFolds(self):
        '''
        check KFolds has been worked correctly.
        sum of validation set from each folds should be same as train set.
        '''
        train_patients = self.readCSV('train')

        fold_num = self.__fold_num
        kfold_patients = []
        kfold_len = []
        for idx in range(1, fold_num+1):
            kfold_train_patients = self.readCSV(phase='train', fold_idx=idx)
            kfold_valid_patients = self.readCSV(phase='valid', fold_idx=idx)
            kfold_len.append(f'{len(set(kfold_train_patients))}/{len(set(kfold_valid_patients))}')
            kfold_patients.extend(kfold_valid_patients)
            assert len(set(kfold_train_patients))+len(set(kfold_valid_patients)) == len(train_patients), \
                "Each fold's data should be same with train set."
        
        assert len(set(train_patients))==len(set(kfold_patients)), \
               "Total KFolds validation set should be same with train set."
        
        print(f'\ntotal train : {len(train_patients)}')
        print(f'kfold valid : {len(kfold_patients)} - {kfold_len}')

    def countDisease(self, total_diseases, train_indices, test_indices):
        '''
        counting # of each disease.
        !!! 'test' will be 'validation' when do KFold !!!
        '''
        diseases_set = {'train':[total_diseases[idx] for idx in train_indices],
                        'test' :[total_diseases[idx] for idx in test_indices]}

        for phase in ['train', 'test']:
            for disease in self.getDiseaseKeys():
                self.setDiseaseDict(phase, disease, diseases_set[phase].count(disease))
            print(f'{phase}-{dict(sorted(self.getDiseaseDict(phase).items(), key=lambda x : x[1], reverse=True))}')
        
    def saveSplitInfo(self, total_patients, train_indices, test_indices, fold_idx=None):
        '''
        save the 'train', 'test' data.
        !!! 'test' will be 'validation' when do KFold !!!
        '''
        if fold_idx ==None:
            split = ['train', 'test']
        else:
            split = ['train', 'valid']

        patients_set = {split[0]:[total_patients[idx] for idx in train_indices],
                        split[1]:[total_patients[idx] for idx in test_indices]}

        for phase in split:
            self.saveInfo(phase, patients_set, fold_idx)

    def saveInfo(self, phase, patients_set, fold_idx=None):
        '''
        if fold_idx == None : train/test(phase) split.
        else : train/valid(phase) split.
        saved data : {'index', 'patient', 'disease'}
        total_path is set right here.
        '''
        d = list(self.getDiseaseKeys())
        d.remove('NORMAL')
        self.__disease_dir = ('_').join(sorted(d))
        if fold_idx == None:
            file_dir = os.path.join('./Data/input',self.__dim,self.__disease_dir,f'test_rate_{self.__test_rate}',phase)
        else:
            file_dir = os.path.join('./Data/input',self.__dim,self.__disease_dir,f'test_rate_{self.__test_rate}','train')
            fold_dir = os.path.join(file_dir, f'{self.__fold_num}fold')

            file_dir = os.path.join(fold_dir, f'fold{fold_idx}')

        os.makedirs(file_dir, exist_ok=True)
            
        file_path = os.path.join(file_dir,f'{phase}.csv')

        df = pd.DataFrame({'patient': patients_set[phase],
                           'disease': [self.getDataDict()[patient] for patient in patients_set[phase]]})

        df.to_csv(file_path, index=True, index_label='index', mode='w')

        self.setTotalPath(phase, file_path, fold_idx)

    def loadLabel(self):
        ''' Load label xlsx file from self.__label_path. '''
        assert self.__input_label_path, "You need to set the label path."
        assert ".xlsx" in self.__input_label_path, "Label data should be .xlsx file."

        return pd.read_excel(self.__input_label_path, engine='openpyxl')[:500]

    def setDiseaseKeys(self):
        '''
        set sorts of diseases. 
        self.__disease_keys = ['NORMAL', 'AMD', 'DR', 'CNV', 'CSC', 'RVO', 'OTHERS']
        '''
        self.__diseases_keys = set(self.getDataDict().values())
    def getDiseaseKeys(self):
        '''
        return sorts of disease(list). 
        ex) ['NORMAL', 'AMD', 'DR', 'CNV', 'CSC', 'RVO', 'OTHERS']
        '''
        return self.__diseases_keys

    def setDiseaseDict(self, phase, disease, number):
        '''
        set disease dictionary.
        phase = 'train/test' or 'train/valid'(for KFolds)
        self.__disease = {phase : {disease: # of disease}}
        '''
        self.__diseases[phase][disease] = number
    def getDiseaseDict(self, phase='train'):
        '''
        phase : train / test
        return disease dictionary : {'train'/'test':{disease: # of disease}}
        '''
        return self.__diseases[phase]

    def setDataDict(self, patient, disease):
        '''
        set data dictionary = {patient:disease}
        '''
        self.__data_dict[int(patient)]=disease
    def getDataDict(self):
        '''
        return data dictionary = {patient:disease}
        '''
        return self.__data_dict

    def getLabel(self):
        '''
        return Label data which type is pandas DataFrame.
        '''
        return self.__label_pdFrame

    def setTotalPath(self, phase, path, fold_idx=None):
        '''
        phase : train, test, kfold
        path : train_path(str), test_path(str), kfold_path(dict)
        '''
        if fold_idx == None:
            self.__total_path[phase]=path
        else:
            if phase == 'train':
                self.__total_path.update({f'fold{fold_idx}':{}})
            self.__total_path[f'fold{fold_idx}'].update({phase:path})
            
    def getTotalPath(self):
        '''
        return total_path('train', 'test', 'fold1,2,...n')
        '''
        return self.__total_path

    def getInputDataPath(self):
        return {'data':self.__input_data_path, 'lable':self.__input_label_path}

    # from this part, all the methods are for the Dataset in pytorch.
    def setDataset(self, phase, fold_idx=None, data_num=None):
        # pre-load dataset.
        assert phase!=None or fold_idx!=None, \
            "Set the phase and fold_idx for getDataset."

        self.setPhase(phase)
        self.set_patients(phase, fold_idx, data_num)

        if self.__dim=='2d':
            self.__X[phase] = list(self.getImage(phase, fold_idx))[0]
        else:
            self.__X[phase] = list(self.getNifti(phase, fold_idx))[0]

        self.__y[phase] = list(self.getOneHot())

    def getDataset(self, phase):
        return self.__X[phase], self.__y[phase]

    def setCurrentData(self, data_dict):
        '''
        data_dict = {patient:disease} of current phase & fold.
        '''
        self.__current_data_dict = data_dict
    def getCurrentData(self):
        return self.__current_data_dict

    def getX(self):
        return self.__X
    def gety(self):    
        return self.__y
    def getPhase(self):
        return self.__phase
    def setPhase(self, phase):
        self.__phase =phase
    def setInputShape(self, input_shape):
        self.__input_shape = input_shape
    def getInputShape(self):
        return self.__input_shape
    def get_options_for_dir_name(self):
        disease = self.__disease_dir
        model = self.__model
        filtering = self.__filter
        fold_num = f'{self.__fold_num}fold'
        flt = 'flt_o' if self.__flatten else 'flt_x'
        classification = 'binary' if self.args.mergeDisease else 'multi'
        options = f'{disease}/{model}/{filtering}/{fold_num}/{flt}/{classification}'
        return options
    
    def setOutputDir(self):
        flt = 'flt_o' if self.__flatten else 'flt_x'
        options = self.get_options_for_dir_name()
        log_dir = os.path.join('./Data/output/log', options)
        check_dir = os.path.join('./Data/output/checkpoint', options)
        result_dir = os.path.join('./Data/output/result', options)
        tb_writer_dir = os.path.join('./Data/output/tensorboard', options)
        best_parameter_dir = os.path.join('./Data/output/best_parameter', options)
        roc_plot_dir = os.path.join('./Data/output/roc_plot', options)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(check_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(tb_writer_dir, exist_ok=True)
        os.makedirs(best_parameter_dir, exist_ok=True)
        os.makedirs(roc_plot_dir, exist_ok=True)

        self.__output_dir = {'log':log_dir,  
                             'checkpoint':check_dir, 
                             'result':result_dir,
                             'tensorboard':tb_writer_dir,
                             'best_parameter':best_parameter_dir,
                             'roc_plot':roc_plot_dir}

    def getOuputDir(self):
        '''
        return output directory : log/checkpoint/result/tensorboard/best_parameter
        '''
        return self.__output_dir

    def __len__(self):
        '''
        Function : get length of datasets.
        Note     : Should be defined for using pytorch Dataloader.
        data_X, data_y should be set.
        '''
        phase = self.getPhase()
        self.len = len(self.getX()[phase])
        return self.len

    def __getitem__(self, idx):
        '''
        Function : get datasets for X/y(=Volume/Label) data with indexing 
        Note     : This method is going to be called by DataLoader in pytorch.
                    So, highly recommended convert data to tensor in this part.
        '''
        phase = self.getPhase()
        X = self.toTensor(self.getX()[phase][idx])
        if self.__pre_train:
            y = self.gety()[phase][idx]
        else:
            y = self.toTensor(self.gety()[phase][idx])

        return list(X), list(y)

    def set_patients(self, phase, fold_idx, data_num=None):
        import random
        if data_num is None:
            self.__patients = sorted(self.readCSV(phase, fold_idx))
            if phase == 'total':
                self.__patients = [p for p in self.__patients if p not in self.__skipList]
        else:
            random.seed(5)
            self.__patients = sorted(random.sample(self.readCSV(phase, fold_idx), data_num))
        # import random
        # patients = self.readCSV(phase, fold_idx)
        # random.shuffle(patients)
        # self.__patients=patients

    def get_patients(self):
        return self.__patients

    def getImage(self, phase, fold_idx):
        '''
        phase = 'train', 'test', f'fold[fold_idx]':{'train', 'valid}'
        load 2D image from path of selected phase.
        '''

        image_list = []
        if 'vgg' in self.args.model.lower():
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))])
        elif 'vit' in self.args.model.lower():
            transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))])
        elif 'res' in self.args.model.lower():
            transform = transforms.Compose([transforms.Resize((304, 304)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))])
        elif 'incept' in self.args.model.lower():
            transform = transforms.Compose([transforms.Resize((299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))])
        current_data_dict = {}
        
        for idx, patient in enumerate(self.get_patients()):
            filename = os.path.join(self.getInputDataPath()['data'], f"{patient}.png")
            im=Image.open(filename)
            if self.is_patch:
                current_data_dict[patient] = self.getDataDict()[patient]
                img_arr = np.asarray(im)
                T = (80*3/np.mean(img_arr))
                img_size = img_arr.shape
                print(f'{patient} og->resized : {img_size}',end='->')
                patch_num = 16 if patient < 10301 else 4
                row_size, col_size = (int(img_size[0]//np.sqrt(patch_num)), int(img_size[1]//np.sqrt(patch_num)))
                print(f'({row_size}, {col_size})')
                image_list_save = []
                cnt = 0
                for row in range(int(np.sqrt(patch_num))):
                    for col in range(int(np.sqrt(patch_num))):
                        img_arr_ = img_arr[(row*row_size):(row*row_size)+row_size, (col*col_size):(col*col_size)+col_size]
                        cnt += 1
                        if patient < 10301:
                            if cnt in [6,7,10,11]:
                                arr = img_arr_
                                # arr_og = img_arr_
                                lower = np.percentile(arr[np.where(arr > 0)], 1)
                                upper = np.percentile(arr[np.where(arr > 0)], 99.94)

                                arr_clip_up_low = np.zeros(np.shape(arr))
                                arr_clip_up_low[(lower <= arr) & (arr <= upper)] = arr[(lower <= arr) & (arr <= upper)]
                                arr_clip_up_low = np.uint8((arr_clip_up_low - arr_clip_up_low.min()) / (arr_clip_up_low.max()-arr_clip_up_low.min())*255)

                                # contrast limit가 2이고 title의 size는 8X8
                                clahe = cv2.createCLAHE(clipLimit=T, tileGridSize=(4,4)) # T=5
                                arr_clip_up_low = clahe.apply(arr_clip_up_low)
                                img_arr_ = arr_clip_up_low
                                pil_img = Image.fromarray(np.uint8(img_arr_))
                                # pil_img = Image.fromarray(np.uint8(arr_og))
                                pil_img_resized = pil_img.resize((100, 100), Image.NEAREST)

                                image_list.append(transform(np.asarray(pil_img_resized)))
                                image_list_save.append(np.asarray(pil_img_resized))
                        else:
                            pil_img = Image.fromarray(np.uint8(img_arr_))
                            pil_img_resized = pil_img.resize((100, 100), Image.NEAREST)

                            image_list.append(transform(np.asarray(pil_img_resized)))
                            image_list_save.append(np.asarray(pil_img_resized))
                
                self.save_patch_img(patient, image_list_save)

            else:
                image_list.append(transform(im))
                current_data_dict[patient] = self.getDataDict()[patient]
            T = 'no patch' if not self.is_patch else T
            print(f"{f'[{idx+1:03d}] image get':16} : {self.getDataDict()[patient]} - ", T)

        self.setInputShape(input_shape = image_list[0].shape)
        self.setCurrentData(current_data_dict)
        yield image_list

    def save_patch_img(self, patient, image_list):
        save_dir = os.path.join(f'{self.data_path}/Patch_0318_P16_CLAHE', f'{patient}')
        # save_dir = os.path.join(f'{self.data_path}/Patch_old_0316', f'{patient}')
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, img in enumerate(image_list):
            save_path = os.path.join(save_dir, f'{patient}_{idx+1}.png')
            img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
            pil_img = Image.fromarray(img)

            pil_img.save(save_path, format='png')


    def getNifti(self, phase, fold_idx):
        niftilist = []
        current_data_dict = {}
        patients = self.get_patients()
        for idx, patient in enumerate(patients):
            niftipath = os.path.join(self.getInputDataPath()['data'], f"{patient}.nii.gz")
            nifti = nib.load(niftipath)
            nifti.uncache()
            niftilist.append(next(self.minMaxNormalize(nifti, patient)))
            current_data_dict[patient] = self.getDataDict()[patient]
            print(f"\r{f'[{idx+1:03d}] nifti get':16} : {self.getDataDict()[patient]}", end='', flush=True)
        print()
        print(f"{'total nifti':16} : {len(niftilist):4}")
        self.setInputShape(input_shape = niftilist[0][np.newaxis, :].shape)
        self.setCurrentData(current_data_dict)
        yield niftilist
    
    def minMaxNormalize(self, volume, patient):
        v = np.asarray(volume.dataobj).transpose(1,0,2)
        print(v.shape) 
        self.__3d_minMax_values[patient]={'max':v.max(), 'min':v.min()}
        v = np.array((v-v.min())/(v.max()-v.min()), dtype='f2')
        yield v

    def toTensor(self, x):
        yield torch.from_numpy(np.asarray(x)).float().to(device=self.__device)

    def setDiseaseLabel(self):
        disease_index = {d:i for i, d in enumerate(sorted(self.getDiseaseKeys()))}
        for d in sorted(self.getDiseaseDict()):
            if d=='NORMAL':
                d2 = [d for d, i in disease_index.items() if i == 0].pop()
                disease_index[d], disease_index[d2] = disease_index[d2], disease_index[d]

        self.__disease_label = disease_index
    
    def getDiseaseLabel(self):
        return self.__disease_label

    def getOneHot(self):
        label2Num = []
        onehot_list = []
        patients = []
        self.setDiseaseLabel()
        table = self.getDiseaseLabel()
        loss = self.__loss
        
        for idx, (patient, disease) in enumerate(self.getCurrentData().items()):
            if loss in ["ce", "fcl", "nll"] : 
                if self.__isMerge == False:
                    if self.__pre_train:
                        label2Num.append([disease, patient])
                    else:
                        label2Num.extend([table[disease]])
                else: 
                    if self.__pre_train:
                        label2Num.extend([int(disease != 'NORMAL'), patient])
                    else:
                        if self.is_patch:
                            patch_num = 4 if patient < 10301 else 4
                            label2Num.extend([int(disease != 'NORMAL')]*patch_num)
                        else:
                            label2Num.extend([int(disease != 'NORMAL')])
                y = label2Num
            else:
                if self.__isMerge ==False:
                    onehot = [0.]*len(table)
                    onehot[table[disease]]=1.
                    onehot_list.append(onehot)
                else:
                    print("!! Set onehot for binary classification !!")
                    pass
                y = onehot_list
            # print(f"\r{f'[{idx+1:03d}] onehot encoding':16} : {patient} - {disease}", end='', flush=True)
        print()
        print(f"{'total label':16} : {len(y):4}")
        # if np.sum(label2Num)!=0. : y = label2Num
        # else : y = onehot_list 
        yield from y

    def sortTable(self, reverse = False):
        if not reverse : 
            table = dict(sorted(self.getDiseaseLabel().items(), key = lambda x : x[1])) 
        else : 
            table = {item:key for key, item in self.getDiseaseLabel().items()}
            table = dict(sorted(table.items(), key = lambda x : x[0])) 
        return table
