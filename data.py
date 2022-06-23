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
from utils.progress_bar import progress_bar

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
        self.__selected_disease = args.disease
        self.__is_merge = args.mergeDisease
        self.__fold_num = args.fold_num
        self.__seed = 304
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
        self.__is_patch = args.patch_mode
        self.__data_path = args.data_path
        self.__grouped_num = 0

    def __call__(self):
        self.set_skip_list()
        self.check_loaded_path()
        self.set_label()
        self.split_train_test()
        self.split_train_valid()
        self.set_output_dir()
        print()

    def set_skip_list(self):
        '''
        The Retinal data can have a noise or an error 
        when they scanned. So, we've check the data
        and remove the crashed or partially broken data.
        
        For 3D Autoencoder pre-training, we need to 
        consider the topological and volumetric structure.
        So, the highly tilted retinal volume data had been excepted.
        '''
        skip = [10035, 10057, 10114, 10219, 10220]
        
        if self.__pre_train:
            
            skip.extend([10039, 10046, 10055, 10074, 10080,
        	             10089,	10111, 10212, 10224, 10257, 
                         10285, 10288])

        self.__skip_list = skip

    def check_loaded_path(self):
        '''
        Checking loading data path.
        Basically, 
        for 2D, the '.png' is extension.
        for 3D, the '.nii' is extension.
        '''
                            
        data_dir = self.__input_data_path
        
        extract_extension = lambda x: ('.').join(x.split('.')[1:])
        extension_list = [extract_extension(x) for x in os.listdir(data_dir)]
        extension = set(extension_list)
        
        if len(extension)>1:
            print("There are different types of data in data_dir.")

        cnt = {k:extension_list.count(k) for k in extension if k != ''}
        
        print(f'Data directory : {data_dir}')
        print(r'{File extension: Numbers} :', f'{cnt}')

    def set_label(self):
        def load_label(label_path):
            ''' 
            Load label xlsx file from self.__label_path. 
            '''
            assert label_path, "You need to set the label path."
            assert ".xlsx" in label_path, "Label data should be .xlsx file."
            return pd.read_excel(label_path, engine='openpyxl')[:500]

        def set_patient_disease(patient, disease):
            '''
            set data dictionary = {patient:disease}
            '''
            if patient not in self.__skip_list:
                if disease in self.__selected_disease:
                    self.__data_dict[int(patient)]=disease 
            else:
                pass
            self.__grouped_num = len(self.__data_dict)
        def set_disease():
            '''
            set sorts of diseases. 
            self.__disease_keys = {'NORMAL', 'AMD', 'DR', 'CNV', 'CSC', 'RVO', 'OTHERS'}
            '''
            self.__diseases_keys = set(self.get_data_dict().values())
            
        label = load_label(self.__input_label_path)
        
        for patient, disease in iter(zip(label['ID'], label['Disease'])):
            set_patient_disease(patient, disease)
            set_disease()

        print(f'Grouped labels : {list(self.__diseases_keys)}')
        print(f'Grouped data number : {self.__grouped_num}')

    def split_train_test(self):
        '''
        This function is splitting the data set to Train/Test.
        Utilizing Stratified Cross Validation, 
        the disease label class will have a even distribution.
        '''
        splits = int(np.ceil(1/self.__test_rate))
        print(f'Test rate(n_splits) : {self.__test_rate}({splits})')
        print(f'{"[ Train/Test Split ]":=^60}')
        
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.__seed)
        patients = list(self.get_data_dict().keys())
        diseases = list(self.get_data_dict().values())
        for train_indices, test_indices in skf.split(patients, diseases):
            self.count_disease(diseases, train_indices, test_indices)
            self.save_split_info(patients, train_indices, test_indices)
            print()
            break

    def split_train_valid(self):
        '''
        Considering the imbalanced and uneven data set,
        The 'Stratified KFold classification' is used.
        '''
        train_path = self.get_total_path()['train']
        train_pdFrame = pd.read_csv(train_path)
        train_patients = list(train_pdFrame['patient'])
        train_disease = list(train_pdFrame['disease'])

        skf = StratifiedKFold(n_splits=self.__fold_num, shuffle=True, random_state=self.__seed)
        fold_idx = 1
        for train_indices, validation_indices in skf.split(train_patients, train_disease):
            print(f'{f"[ fold {fold_idx} ]":=^60}')
            self.count_disease(train_disease, train_indices, validation_indices)
            self.save_split_info(train_patients, train_indices, validation_indices, fold_idx)
            fold_idx+=1

        self.check_folds()
        print()

    def set_output_dir(self):
        '''
        Set the output directory path.
        Based on the arguements that you've set.
        The 'self.get_options()' will return that oprtion.
        '''
        options = self.get_options()
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

    def read_csv(self, phase, fold_idx=None):
        '''
        Read the label csv file.
        This csv file has patients lists and their disease info.
        For Kfold cross validation,
        Each fold has been saved seperately for reproducibility of result.
        '''
        if fold_idx==None:
            if phase == 'total':
                self.set_total_path('total', './Data/input/total.csv')
            path = self.get_total_path()[phase]
        else:
            path = self.get_total_path()[f'fold{fold_idx}'][phase]
        
        pdFrame = pd.read_csv(path)
        data_dict = list(pdFrame['patient'])
        return data_dict

    def check_folds(self):
        '''
        Check KFolds has been worked correctly.
        Load saved CSV data labels and check the ratio and the # of data.
        Sum of validation set from each folds should be same as train set.
        '''
        train_patients = self.read_csv('train')

        fold_num = self.__fold_num
        kfold_patients = []
        kfold_len = []
        for idx in range(1, fold_num+1):
            kfold_train_patients = self.read_csv(phase='train', fold_idx=idx)
            kfold_valid_patients = self.read_csv(phase='valid', fold_idx=idx)
            kfold_len.append(f'{len(set(kfold_train_patients))}/{len(set(kfold_valid_patients))}')
            kfold_patients.extend(kfold_valid_patients)
            assert len(set(kfold_train_patients))+len(set(kfold_valid_patients)) == len(train_patients), \
                "Each fold's data should be same with train set."
        
        assert len(set(train_patients))==len(set(kfold_patients)), \
               "Total KFolds validation set should be same with train set."
        
        train_num = len(train_patients)
        print(f'\nTotal - train, test : {train_num}, {self.__grouped_num-train_num}')
        print(f'Folds - train/valid : {len(kfold_patients)} - {kfold_len}')

    def count_disease(self, total_diseases, train_indices, test_indices):
        '''
        Counting # of each disease.
        Basically, split train/test, when train is working,
        trainset needs to split into train/valid.
        '''
        diseases_set = {'train':[total_diseases[idx] for idx in train_indices],
                        'test' :[total_diseases[idx] for idx in test_indices]}

        for phase in ['train', 'test']:
            for disease in self.get_disease_keys():
                self.set_disease_dict(phase, disease, diseases_set[phase].count(disease))
            print(f'{phase}-{dict(sorted(self.get_disease_dict(phase).items(), key=lambda x : x[1], reverse=True))}')
        
    def save_split_info(self, total_patients, train_indices, test_indices, fold_idx=None):
        '''
        save the 'train', 'test' data.
        Total Data -> 'Train / Test' (when fold_idx=None)
        Train Data -> 'Train / Validation' (when fold_idx!=None)
        '''
        if fold_idx ==None:
            split = ['train', 'test']
        else:
            split = ['train', 'valid']

        patients_set = {split[0]:[total_patients[idx] for idx in train_indices],
                        split[1]:[total_patients[idx] for idx in test_indices]}

        for phase in split:
            self.save_info(phase, patients_set, fold_idx)

    def save_info(self, phase, patients_set, fold_idx=None):
        '''
        This program is operated with label data.
        We don't call all the volume or image data before training.
        So, we save the split-info to the txt-file.
        Save data : {'index', 'patient', 'disease'}
        Save dir = './Data/input'
        Save path is based on below features.
        Set the self.__total_path based on the path either.
        
        [features]
        if fold_idx == None : 
            train/test(phase) split.
        else : 
            train/valid(phase) split.
        '''
        
        d = list(self.get_disease_keys())
        d.remove('NORMAL')
        self.__disease_dir = ('_').join(sorted(d))
        if fold_idx == None:
            file_dir = os.path.join('./Data/input',
                                    self.__dim, self.__disease_dir,
                                    f'test_rate_{self.__test_rate}',phase)
        else:
            file_dir = os.path.join('./Data/input',
                                    self.__dim, self.__disease_dir,
                                    f'test_rate_{self.__test_rate}','train')
            fold_dir = os.path.join(file_dir, f'{self.__fold_num}fold')
            file_dir = os.path.join(fold_dir, f'fold{fold_idx}')
            
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir,f'{phase}.csv')
        df = pd.DataFrame({'patient': patients_set[phase],
                           'disease': [self.get_data_dict()[patient] for patient in patients_set[phase]]})
        df.to_csv(file_path, index=True, index_label='index', mode='w')
        
        self.set_total_path(phase, file_path, fold_idx)

    def get_disease_keys(self):
        '''
        return sorts of disease(list). 
        ex) ['NORMAL', 'AMD', 'DR', 'CNV', 'CSC', 'RVO', 'OTHERS']
        '''
        return self.__diseases_keys

    def set_disease_dict(self, phase, disease, number):
        '''
        set disease dictionary.
        phase = 'train/test' or 'train/valid'(for KFolds)
        self.__disease = {phase : {disease: # of disease}}
        '''
        self.__diseases[phase][disease] = number
    
    def get_disease_dict(self, phase='train'):
        '''
        phase : train / test
        return disease dictionary : {'train'/'test':{disease: # of disease}}
        '''
        return self.__diseases[phase]

    def get_data_dict(self):
        '''
        return {patient:disease}
        '''
        return self.__data_dict

    def set_total_path(self, phase, path, fold_idx=None):
        '''
        phase : train / test / kfold
        path : train_path(str), test_path(str), kfold_path(dict)
        fold_idx : 'train/test' if None else 'train/valid'
        '''
        if fold_idx == None:
            self.__total_path[phase]=path
        else:
            if phase == 'train':
                self.__total_path.update({f'fold{fold_idx}':{}})
            self.__total_path[f'fold{fold_idx}'].update({phase:path})

    def get_total_path(self):
        '''
        return total_path('train', 'test', 'fold1,2,...n')
        '''
        return self.__total_path

    def get_options(self):
        '''
        Set the options for the output directory.
        Base on the arguments that you set.
        '''
        disease = self.__disease_dir
        model = self.__model
        filtering = self.__filter
        fold_num = f'{self.__fold_num}fold'
        flt = 'flt_o' if self.__flatten else 'flt_x'
        classification = 'binary' if self.__is_merge else 'multi'
        options = f'{disease}/{model}/{filtering}/{fold_num}/{flt}/{classification}'
        return options

    def get_input_path(self):
        '''
        Return the input_data info which is
        data_path <--> label(patient, disease)
        '''
        return {'data':self.__input_data_path, 'lable':self.__input_label_path}

    # from this part, all the methods are for the Dataset in pytorch.
    def set_dataset(self, phase, fold_idx=None, data_num=None):
        '''
        Customize function for the Torch-Dataset.
        [Data X]
        For 2D, get Image data(2d array) from saved dir.
        For 3D, get Nifti data(3d array) from saved dir.
        [Data y]
        Load the label data and set them for the y.
        This will use the 'one-hot encoding'.
        '''
        # pre-load dataset.
        assert phase!=None or fold_idx!=None,\
               "Set the phase and fold_idx for get_dataset."

        self.set_phase(phase)
        self.set_patients(phase, fold_idx, data_num)

        if self.__dim=='2d':
            self.__X[phase] = list(self.get_image(phase, fold_idx))[0]
        else:
            self.__X[phase] = list(self.getNifti(phase, fold_idx))[0]

        self.__y[phase] = list(self.getOneHot())

    def get_dataset(self, phase):
        return self.__X[phase], self.__y[phase]

    def set_current_data(self, data_dict):
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
    
    def set_phase(self, phase):
        self.__phase =phase
        
    def set_input_shape(self, input_shape):
        self.__input_shape = input_shape
        
    def getInputShape(self):
        return self.__input_shape

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
        '''
        This function is loading the saved fold.csv and set the patients.
        Load : each fold patients & disease list - './Data/input/...'
        Set  : self.__patients = [10001, 10002, ... ]
        '''
        import random
        if data_num is None:
            self.__patients = sorted(self.read_csv(phase, fold_idx))
            if phase == 'total':
                self.__patients = [p for p in self.__patients if p not in self.__skip_list]
        else:
            random.seed(5)
            self.__patients = sorted(random.sample(self.read_csv(phase, fold_idx), data_num))

    def get_patients(self):
        '''
        return self.__patients = [10001, 10002, ... ]
        '''
        return self.__patients

    def get_image(self, phase, fold_idx):
        '''
        phase = 'train', 'test', f'fold[fold_idx]':{'train', 'valid}'
        load 2D image from path of selected phase.
        '''
        def get_transformer():
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
            return transform
        
        def image_resize_info(patient, image_size):
            '''
            FOV66 - 400x400 => patch_num 16
            FOV33 - 304x304 => patch_num 4
            to make them similar resolutions.
            The row, col size are defined by patch_num respectively.
            '''
            patch_num = 4
            row_size = int(image_size[0]//np.sqrt(patch_num))
            col_size = int(image_size[1]//np.sqrt(patch_num))
            print(f'{patient} og->resized : {image_size}',end='->')
            return row_size, col_size, patch_num
        
        def image_clipping(image_arr):
            '''
            Clipping set min/max intensity has matched to
            specific intensity value from percentage of total
            intensity. The reason we use this process is when
            we adap min-max normalize without this, the 255 could
            be the max and this value could be the noise.
            So for proper min-max normalization, we use this process.
            '''
            arr = image_arr.copy()
            lower = np.percentile(arr[np.where(arr > 0)], 5)
            upper = np.percentile(arr[np.where(arr > 0)], 99.9)

            arr_clip_up_low = image_arr.copy()
            arr_clip_up_low[lower>arr_clip_up_low] = 0
            arr_clip_up_low[upper<arr_clip_up_low] = int(upper)
            
            # # min-max
            # arr_clip_up_low = np.uint8((arr_clip_up_low - arr_clip_up_low.min()) / 
            #                             (arr_clip_up_low.max()-arr_clip_up_low.min())*255)
            return arr_clip_up_low
        
        def set_resized_image_list(image_arr, image_list_save):
            '''
            Resizing by utilizing PIL library.
            The interpolation is NEAREST. 
            '''
            pil_img = Image.fromarray(np.uint8(image_arr))
            pil_img_resized = pil_img.resize((100, 100), Image.NEAREST)

            image_list.append(transform(np.asarray(pil_img_resized)))
            image_list_save.append(np.asarray(pil_img_resized))
            return image_list_save
        
        def get_patch_images(patient, image):
            '''
            When self.__patch_mode = True,
            This function will split the image to 4 images.
            '''
            current_data_dict[patient] = self.get_data_dict()[patient]
            img_arr = np.asarray(image) if patient > 10300 else np.asarray(image)[100:300, 100:300]
            img_size = img_arr.shape
            row_size, col_size, patch_num = image_resize_info(patient, img_size)
            # threshold : utilizing img_arr to adapt 'each patient'
            threshold = (256/np.mean(img_arr))
            image_list_save = []
            cnt = 0
            
            # FOV66 needs to be FOV33 -> Center cropping.
            for row in range(int(np.sqrt(patch_num))):
                for col in range(int(np.sqrt(patch_num))):
                    # crop patch.
                    img_arr_ = img_arr[(row*row_size):(row*row_size)+row_size,
                                       (col*col_size):(col*col_size)+col_size]
                    cnt += 1
                    if patient < 10301:
                        '''
                        The FOV66(10001~10300) image has different
                        contrast from FOV33(10301~10500).
                        This process is needed to improve the performace.
                        Control the contrast : 
                        CLAHE, threshold -> base on the each patient image property.
                        '''
                        # if cnt in [6,7,10,11]: # center 4 Images.
                        arr_clip_up_low = image_clipping(img_arr_)
                        # Contrast control - CLAHE
                        clahe = cv2.createCLAHE(clipLimit=threshold, tileGridSize=(4, 4))
                        arr_clip_up_low = clahe.apply(arr_clip_up_low)
                        image_list_save = set_resized_image_list(arr_clip_up_low, image_list_save)
                    else:
                        image_list_save = set_resized_image_list(img_arr_, image_list_save)

            return image_list_save, threshold
        
        def save_patch_img(patient, image_list):
            '''
            Gathered 4 images of 1 patient image is saved here.
            Save dir name is set by today's date(YYYY_MMDD).
            '''
            save_dir = os.path.join(f'{self.__data_path}/{self.args.date}_P16_CLAHE', f'{patient}')
            os.makedirs(save_dir, exist_ok=True)
            
            for idx, img in enumerate(image_list):
                save_path = os.path.join(save_dir, f'{patient}_{idx+1}.png')
                # img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
                pil_img = Image.fromarray(img)

                pil_img.save(save_path, format='png')
        
        current_data_dict = {}
        image_list = []
        transform = get_transformer()
        
        for idx, patient in enumerate(self.get_patients()):
            filename = os.path.join(self.get_input_path()['data'], f"{patient}.png")
            image=Image.open(filename)
            if self.__is_patch:
                image_list, threshold = get_patch_images(patient, image)
                save_patch_img(patient, image_list)
                image_list = [ transform(i) for i in image_list ]
            else:
                image_list.append(transform(image))
                current_data_dict[patient] = self.get_data_dict()[patient]
            resized_image_shape = image_list[0].shape
            T = 'no patch' if not self.__is_patch else threshold
            print(f'{np.multiply(resized_image_shape, 2)}')
            print(f"{f'[{idx+1:03d}] image get':16} : {self.get_data_dict()[patient]} - ", T)

        self.set_input_shape(resized_image_shape)
        self.set_current_data(current_data_dict)
        yield image_list

    def getNifti(self, phase, fold_idx):
        niftilist = []
        current_data_dict = {}
        patients = self.get_patients()
        for idx, patient in enumerate(patients):
            niftipath = os.path.join(self.get_input_path()['data'], f"{patient}.nii.gz")
            nifti = nib.load(niftipath)
            nifti.uncache()
            niftilist.append(next(self.minMaxNormalize(nifti, patient)))
            current_data_dict[patient] = self.get_data_dict()[patient]
            print(f"\r{f'[{idx+1:03d}] nifti get':16} : {self.get_data_dict()[patient]}", end='', flush=True)
        print()
        print(f"{'total nifti':16} : {len(niftilist):4}")
        self.set_input_shape(niftilist[0][np.newaxis, :].shape)
        self.set_current_data(current_data_dict)
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
        disease_index = {d:i for i, d in enumerate(sorted(self.get_disease_keys()))}
        for d in sorted(self.get_disease_dict()):
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
                if self.__is_merge == False:
                    if self.__pre_train:
                        label2Num.append([disease, patient])
                    else:
                        label2Num.extend([table[disease]])
                else: 
                    if self.__pre_train:
                        label2Num.extend([int(disease != 'NORMAL'), patient])
                    else:
                        if self.__is_patch:
                            patch_num = 4 if patient < 10301 else 4
                            label2Num.extend([int(disease != 'NORMAL')]*patch_num)
                        else:
                            label2Num.extend([int(disease != 'NORMAL')])
                y = label2Num
            else:
                if self.__is_merge ==False:
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
