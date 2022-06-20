import os 
from os      import mkdir
from os.path import join, isdir

class setDirectory():
    def __init__(self, *dirs):
        self.base_dirs = dirs
        self.dimension = ['2d','3d']
        self.model     = ['clinica_cnn3d', 'vgg16_3d', 'Res50_3d']
        self.filter    = ['OG','Curvelet']
        self.status    = ['Train','Valid','Best','Test']
        self.baseDirCheck()
        self.setFullDir()
        

    def __call__(self, args):        
        dirs = self.getSelectedDirs(args)
        self.setModelFoldDir(dirs)
        return dirs

    def baseDirCheck(self):
        for base_dir in self.base_dirs:
            base_dir_name = base_dir.split('/')[-1]
            if not isdir(base_dir):
                mkdir(base_dir)
                print(f"{base_dir_name} has been made.")
            else : pass
                

    def setFullDir(self):
        for base in self.base_dirs:
            for dimension in self.dimension:
                for model in self.model:
                    for filtername in self.filter:
                        if 'check' in base : self.status = ['train','best']
                        else: self.status = ['train','valid','test','best']
                        for status in self.status:
                            dirpath = join(base, dimension, model, filtername, status)
                            os.makedirs(dirpath, exist_ok=True)

    def setModelFoldDir(self, dirs):
        model_train_dir  = os.path.join(dirs['checkpoint'], 'train')
        model_best_dir   = os.path.join(dirs['checkpoint'],  'best')
        result_train_dir = os.path.join(dirs['result'],'train')
        result_valid_dir = os.path.join(dirs['result'],'valid')
        result_test_dir  = os.path.join(dirs['result'], 'test')
        result_best_dir  = os.path.join(dirs['result'], 'best')
        
        for fold_idx in range(self.fold_num):
            model_train_f_dir  = os.path.join(model_train_dir,   "fold"+str(fold_idx+1))
            model_best_f_dir   = os.path.join(model_best_dir,    "fold"+str(fold_idx+1))
            os.makedirs(model_train_f_dir, exist_ok=True)
            os.makedirs(model_best_f_dir , exist_ok=True)
            for ae in ['ae_x','ae_o']:
                result_train_f_dir = os.path.join(result_train_dir,f"{ae}/fold{fold_idx+1}")
                result_valid_f_dir = os.path.join(result_valid_dir,f"{ae}/fold{fold_idx+1}")
                result_best_f_dir  = os.path.join(result_best_dir,f"{ae}/fold{fold_idx+1}")
                os.makedirs(result_train_f_dir , exist_ok=True)
                os.makedirs(result_valid_f_dir , exist_ok=True)
                os.makedirs(result_best_f_dir  , exist_ok=True)
            
            
        
    def getSelectedDirs(self, args):
        self.fold_num = args.fold_num
        dimension, model, filtername = args.dimension, args.model, args.filter
        dirs = {'data':"", 'checkpoint':"", 'result':""}
        for base_dir in self.base_dirs:
            dirpath = join(base_dir, dimension, model, filtername)
            if   'check'    in base_dir : dirs['checkpoint'] = dirpath
            elif 'result'   in base_dir : dirs['result']     = dirpath
            else : dirs['data'] = self.getDataDir(dimension, filtername)
        return dirs
    
    def getDataDir(self, selected_dimension, selected_filter):
        if selected_dimension == '2d' : 
            if selected_filter   == 'OG'       : return "./data/dataset/og"
            elif selected_filter == 'Curvelet' : return "./data/dataset/cvlt"
            else: raise ValueError("Select correct filter name : OG/Curvelet")

        elif selected_dimension == '3d' : 
            if selected_filter   == 'OG'       : return "./data/Nifti/In/FOV_66/SRL"
            elif selected_filter == 'Curvelet' : return "./data/Nifti/In/FOV_66/Curvelet_SRL"
            else: raise ValueError("Select correct filter name : OG/Curvelet")

        else:
            raise ValueError("Worng Dimension")


       
    