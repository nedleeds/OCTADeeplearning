import torch
import numpy as np

SEED = 7

def seed_everything(seed):
    import torch, random, os, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

seed_everything(SEED)

from sklearn.metrics import confusion_matrix, classification_report
class checking():
    def __init__(self, labels=None, cm=None, lss='ce', isMerge=False):
        '''
        cm  : confusion matrix
        lss - loss name 
            : 'MSELoss'-> 'mse', 'CELoss'->'ce', 'FocalLoss'->'fcl'
        '''
        self.cm        = cm
        self.gt_list   = []
        self.pd_list   = []
        self.loss_name = lss        
        self.is_merge = isMerge
        self.is_binary = True if len(labels)==2 else False

        if labels=='None':
            raise ValueError("!!! Set label_table !!!\n[[{disease:index}],[{index:disease}]]") 
        else:
            if self.is_merge:
                self.disease_idx = {'NORMAL':0, 'ABNORMAL':1}
                self.idx_disease = {0:'NORMAL', 1:'ABNORMAL'}
            else:
                self.disease_idx = labels
                self.idx_disease = {idx:disease for disease, idx in labels.items()}
            self.labels = list(self.disease_idx.keys())
            self.classes = len(self.labels)

    def Step(self, gt, pd):
        if type(gt)==list and type(pd)==list: 
            gt = np.asarray(gt)
            pd = np.asarray(pd)
            
        # metric = self.getMetric(gt, pd.argmax(axis=1))
        metric = self.getMetric(gt, pd)
        # self.showMetric(metric)
        

    def Epoch(self, gt, pd):
        if type(gt)==list and type(pd)==list: 
            gt = np.asarray(gt)
            pd = np.asarray(pd)

        # metric = self.getMetric(gt, pd.argmax(axis=1))
        metric = self.getMetric(gt, pd)
        self.showMetric(metric)
        return metric

    def showMetric(self, score):
        if self.is_binary:
            if type(score)==tuple:
                score = score[0]
            print(f"TP/TN/FP/FN : {score['TP']}/{score['TN']}/{score['FP']}/{score['FN']}")
            print(f"{'RCL' :4}:{score['RCL']:>.5f}",end='\t')
            print(f"{'PCS' :4}:{score['PCS']:>.5f}",end='\t')
            print(f"{'F1'  :4}:{score['F1']:>.5f}" ,end='\n')
            print(f"{'BA'  :4}:{score['BA']:>.5f}" ,end='\t')
            print(f"{'SP'  :4}:{score['SP'] :>.5f}",end='\t')
            print(f"{'ACC' :4}:{score['ACC']:>.5f}",end='\n')
        else:
            if type(score) == tuple:
                score = score[0]
            for k in score:
                if k!='accuracy' and k!='loss':
                    print(f'{k:12} : ', end='')
                try:
                    for metric in score[k]:
                        if metric == 'support':
                            print(f'num-{score[k][metric]:3}', end=' ')
                        else:
                            print(f'{metric}-{np.round(score[k][metric], 6):8}', end=' ')
                    print()
                except:
                    print(f'{k:12} : {np.round(score[k], 6)}')

    def showResult(self, gt, pd):
        gt_list, pd_list = [], []
        for gt_idx, pd_idx in zip(gt, pd.argmax(axis=1)):
            gt_list.append(self.idx_disease[gt_idx])
            pd_list.append(self.idx_disease[pd_idx])
        print(f"gt-{gt_list}\npd-{pd_list}")
        print()

    def getMetric(self, gt, pd):
        gt_list, pd_list = [], []
        diseases = list(self.disease_idx.keys())

        for gt_idx, pd_idx in zip(gt, pd):
            gt_list.append(self.idx_disease[gt_idx])
            pd_list.append(self.idx_disease[pd_idx])

        if self.is_merge and self.is_binary:
            # for Binary Class
            cnfsmtrx = confusion_matrix(gt_list, pd_list, labels=diseases)
            result   = classification_report(gt_list, pd_list, labels=diseases, zero_division=0, output_dict=True)
            result   = self.getEvaluate(cnfsmtrx, result)
            cnfsmtrx = None
        elif not self.is_binary:
            cnfsmtrx = confusion_matrix(gt_list, pd_list, labels=diseases)
            result   = classification_report(gt_list, pd_list, labels=diseases, zero_division=0, output_dict=True)
        
        return result, cnfsmtrx

    def getEvaluate(self, cm, result):
        metric = {
            'TP' : 0,   'TN' : 0,   'FP' : 0,   'FN' : 0,
            'BA' : 0.,  'SE' : 0.,  'SP' : 0.,  'ACC': 0,
            'F1' : 0.,  'PCS': 0.,  'RCL': 0.
        }

        if self.classes==2: 
            metric['TN'] = cm.ravel()[0].astype(int)
            metric['FP'] = cm.ravel()[1].astype(int)
            metric['FN'] = cm.ravel()[2].astype(int)
            metric['TP'] = cm.ravel()[3].astype(int)

        if (metric['TP']+metric['FN']): 
            metric['SE'] = np.round(metric['TP']/(metric['TP']+metric['FN']),6)
        else : 
            metric['SE'] = 0

        if (metric['TN']+metric['FP']): 
            metric['SP'] = np.round(metric['TN']/(metric['TN']+metric['FP']),6)
        else : 
            metric['SP'] = 0
        
        metric['BA']  = np.round((metric['SE']+metric['SP'])/2,6)
        metric['F1']  = np.round(result[f"{self.idx_disease[1]}"]['f1-score'],6)
        metric['RCL'] = np.round(result[f"{self.idx_disease[1]}"]['recall']  ,6)
        metric['PCS'] = np.round(result[f"{self.idx_disease[1]}"]['precision'],6)
        metric['ACC'] = np.round((metric['TP']+metric['TN'])/(metric['TP']+metric['TN']+metric['FP']+metric['FN']),6)
    
        return metric