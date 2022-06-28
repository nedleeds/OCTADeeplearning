import os
import collections
from pyexpat import model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from numpy import interp
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from utils.evaluate import checking

def dict_value_append(src_dict, tgt_dict):    
    for k in tgt_dict:
        tgt_dict[k].append(src_dict[k])
    return tgt_dict

def dict_mean_std(dictionary):
    mean_dict = {}
    std_dict = {}
    for k in dictionary:
        mean_dict['mean_'+k] = np.mean(dictionary[k]).round(5)
        std_dict['std_'+k] = np.std(dictionary[k]).round(3)
    
    return mean_dict, std_dict

def kfold_roc_plot(key, config, phase, total_out):
    plot_name = get_plot_name(key)
    fig1 = plt.figure(figsize=[12,12])
    FONT_SIZE = 20
    plt.rc('font', size=FONT_SIZE) # controls default text sizes 
    plt.rc('axes', titlesize=FONT_SIZE) # fontsize of the axes title 
    plt.rc('axes', labelsize=FONT_SIZE) # fontsize of the x and y labels 
    plt.rc('xtick', labelsize=FONT_SIZE) # fontsize of the tick labels 
    plt.rc('ytick', labelsize=FONT_SIZE) # fontsize of the tick labels 
    plt.rc('legend', fontsize=FONT_SIZE) # legend fontsize 
    plt.rc('figure', titlesize=FONT_SIZE) # fontsize of the figure title

    # ax1 = fig1.add_subplot(111,aspect = 'equal')
    # ax1.add_patch(
    #     patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5)
    #     )
    # ax1.add_patch(
    #     patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5)
    #     )

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)

    fold_num = 5
    check = checking(labels = ['NORMAL', 'ABNORMAL'],isMerge = True)
    for fold_idx in range(1, fold_num+1):
        y, y_pred = read_gt_pd(fold_idx, key, config, phase)
        fpr, tpr, t = roc_curve(y, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        total_out['valid']['gt'].extend(y)
        total_out['valid']['pd'].extend(y_pred)
        result, cfmx = check.Epoch(y,y_pred)
        if fold_idx==1:
            results = {k:[] for k in result.keys()}
        results = dict_value_append(result, results)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='fold %d ROC (AUC = %0.5f)' % (fold_idx, roc_auc))    
    
    results_mean, results_std = dict_mean_std(results)
    
    plt.plot([0,1], [0,1],linestyle = '--',lw = 2,color = 'gray')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
            label=r'Mean ROC (AUC = %0.5f)' % (mean_auc),lw=2, alpha=1)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'[{plot_name}] 5-Fold ROC Plot', size=FONT_SIZE, pad=FONT_SIZE)
    plt.legend(loc="lower right")
    # plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    # plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    plt.show()
    roc_path = os.path.join(config['roc_dir'], f'{key}_kfold.png')
    plt.savefig(roc_path)
    return mean_tpr, mean_fpr, mean_auc, total_out, results_mean, results_std

def get_plot_name(key):
    dimension, pre_processing, pre_train, model_name = key.split('_')
    
    model_name = ('').join(model_name.split('-'))
    model_name = model_name+'+' if pre_processing == 'flt-o' else model_name
    pre_train = 'pre-trained' if pre_train == 'ae-o' else ''
        
    return f'{dimension.upper()} {model_name} {pre_train}'
        

def read_gt_pd(fold_idx, key, config, phase):
    if 'test' not in phase:
        phase = 'valid'
    file_dir = config['roc_dir']
    os.makedirs(file_dir, exist_ok=True)
    key = ('_').join(key.split('_')[:-1]).replace('-', '_')
    if fold_idx is not None:
        file_name = f"{key}_{phase}_f{fold_idx}.csv"
    else:
        file_name = f"{key}_{phase}.csv" # for test
    file_path = os.path.join(file_dir, file_name)

    pdFrame = pd.read_csv(file_path)
    y = list(pdFrame['y'])
    y_pred = list(pdFrame['y_pred'])
    return y, y_pred

def total_mean_plot(total_tpr, total_fpr, total_auc, configs):
    FONT_SIZE = 20
    PAD_SIZE = 10
    plt.rc('font', size=FONT_SIZE) # controls default text sizes 
    plt.rc('axes', titlesize=FONT_SIZE) # fontsize of the axes title 
    plt.rc('axes', labelsize=FONT_SIZE) # fontsize of the x and y labels 
    plt.rc('xtick', labelsize=FONT_SIZE) # fontsize of the tick labels 
    plt.rc('ytick', labelsize=FONT_SIZE) # fontsize of the tick labels 
    plt.rc('legend', fontsize=FONT_SIZE*.9) # legend fontsize 
    plt.rc('figure', titlesize=FONT_SIZE) # fontsize of the figure title
    plt.figure(figsize=(12, 8)).clf()
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'gray')

    for key, _ in configs.items():
        legend = get_plot_name(key)
        mean_tpr, mean_fpr, mean_auc = total_tpr[key], total_fpr[key], total_auc[key]
        plt.plot(mean_fpr, mean_tpr,
                 label=f'{legend} (AUC = {mean_auc:0.5f} )',
                 lw=2, alpha=1)
    
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"5-Fold Cross Validation ROC Plot", size=FONT_SIZE, pad=PAD_SIZE)

    roc_dir = './Data/output/roc_plot/AMD_CSC_DR_RVO'
    roc_name = 'valid_mean_roc.png'
    roc_path = os.path.join(roc_dir, roc_name)
    plt.savefig(roc_path)

def total_test_plot(configs, total_out):
    FONT_SIZE = 20
    PAD_SIZE = 10
    plt.rc('font', size=FONT_SIZE) # controls default text sizes 
    plt.rc('axes', titlesize=FONT_SIZE) # fontsize of the axes title 
    plt.rc('axes', labelsize=FONT_SIZE) # fontsize of the x and y labels 
    plt.rc('xtick', labelsize=FONT_SIZE) # fontsize of the tick labels 
    plt.rc('ytick', labelsize=FONT_SIZE) # fontsize of the tick labels 
    plt.rc('legend', fontsize=FONT_SIZE*.9) # legend fontsize 
    plt.rc('figure', titlesize=FONT_SIZE) # fontsize of the figure title
    plt.figure(figsize=(12, 8)).clf()
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'gray')

    for k, c in configs.items():
        name = get_plot_name(k)
        y, y_pred = read_gt_pd(None, key=k, config=c, phase='test')
        total_out[name]['test']['gt'].extend(y)
        total_out[name]['test']['pd'].extend(y_pred)
        fpr, tpr, t = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 
                 label=f'{name} (AUC = {roc_auc:0.5f})', 
                ) 
    
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Test ROC Plot of 5folds", size=FONT_SIZE, pad=PAD_SIZE)

    roc_dir = './Data/output/roc_plot/AMD_CSC_DR_RVO'
    roc_name = 'test_mean_roc.png'
    roc_path = os.path.join(roc_dir, roc_name)
    plt.savefig(roc_path)

    return total_out

def bar_plot(total_out, mean, std):        
    FONT_SIZE = 20 
    PAD_SIZE = 10
    for phase in ['valid', 'test']:
        print()
        check = checking(labels = ['NORMAL', 'ABNORMAL'],isMerge = True)
        scores = []
        scores2 = []
        # get the DataFrame from dictionary 
        xlabels = []
    
        if phase == 'valid':
            for model_name in list(total_out.keys()):
                result = {}
                result2 = {}
                # xlabels.append(model_name.replace('+ ', '+\n'))
                xlabels.append(model_name.replace(' ', '\n'))
                for k1, k2 in zip(mean[model_name][phase], std[model_name][phase]):
                    result[k1.split('_')[-1]] = mean[model_name][phase][k1]
                    result2[k2.split('_')[-1]] = std[model_name][phase][k2]
                result.update({'CNN Models':model_name})
                result2.update({'CNN Models':model_name})
                scores.append(result) 
                scores2.append(result2)
        else:
            # total_gt and total_pd
            for model_name in list(total_out.keys()):
                print(f'{model_name} [{phase}]', end=' -> ')
                xlabels.append(model_name.replace(' ', '\n'))
                # xlabels.append(model_name.replace('+ ', '+\n'))
                result, cfmx = check.Epoch(total_out[model_name][phase]['gt'], total_out[model_name][phase]['pd'])
                result.update({'CNN Models':model_name})
                scores.append(result)

        df=pd.DataFrame(scores)
        df_std = pd.DataFrame(scores2)
        print('mean', '\n', df)
        print('std', '\n', df_std)
        # plot
        sns.set_theme()
        bplot = df.plot(x="CNN Models", y=["F1", "ACC", "BA"], kind="bar", figsize=(12, 8), rot=0, width=0.7)
        plt.ylim([0.6, 0.97])
        plt.xlabel('', fontsize=FONT_SIZE, labelpad =PAD_SIZE) # for transfer learning results
        # plt.xlabel('', fontsize=FONT_SIZE*.8, labelpad =PAD_SIZE) # for all the models
        plt.ylabel('Scores', fontsize=FONT_SIZE, labelpad =PAD_SIZE)
        # bplot.set_xticklabels(labels=xlabels, fontsize=FONT_SIZE) # for transfer learning results
        bplot.set_xticklabels(labels=xlabels, fontsize=FONT_SIZE*.8) # for all the models
        plt.yticks(fontsize=FONT_SIZE)
        title_name = '5-Fold Average Cross Validation Bar Plot' if phase == 'valid' else 'Test Bar Plot'
        # plt.title(f'{title_name}', fontsize=FONT_SIZE, pad=25) # for transfer learning results
        plt.title(f'{title_name}', fontsize=FONT_SIZE, pad=40)
        plt.legend(
                   labels=['F1-score', 'Accuraccy', 'Balanced Accuracy'],
                   bbox_to_anchor=(0, 1.005, 1, 0.2), 
                #    bbox_to_anchor=(0, 0.95, 1, 0.2), # for transfer learning results
                   loc="lower left",
                   fontsize=FONT_SIZE*.8,
                   mode="expand", 
                   borderaxespad=0, 
                   ncol=3
                   )

        y_list = df[['F1', 'ACC', 'BA']].values.reshape(-1)
        for p in bplot.patches:
            bplot.annotate(format(p.get_height(), '.5f'), 
                    # (p.get_x() + p.get_width()/2. +.01, p.get_height()*.95), ## for transfer learning results
                    # size = FONT_SIZE, ## for transfer learning results
                    (p.get_x() + p.get_width()/2. +.01, p.get_height()*.97), ## for all the models
                    size = FONT_SIZE*.8, ## for all the models
                    ha = 'center', 
                    va = 'center', 
                    rotation=90,
                    xytext = (0, -12), 
                    color = 'white',
                    weight="bold",
                    textcoords = 'offset points')
        plt.show()
        
        # save plotted figure
        roc_dir = './Data/output/roc_plot/AMD_CSC_DR_RVO'
        roc_name = f'{phase}_bar.png'
        roc_path = os.path.join(roc_dir, roc_name)
        plt.savefig(roc_path)
        

def main():
    configs = {
               '2d_flt-x_ae-x_Res-152': {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_152_2D/OG/5fold/flt_x/binary/ae_x'},
               '2d_flt-x_ae-x_Res-50': {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_50_2D/OG/5fold/flt_x/binary/ae_x'},
               '2d_flt-x_ae-x_VGG-19': {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/VGG_19_2D/OG/5fold/flt_x/binary/ae_x'},
               '2d_flt-x_ae-x_Incept-V3': {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Incept_v3_2D/OG/5fold/flt_x/binary/ae_x'},
               '3d_flt-o_ae-x_Incept-V3' : {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Incept_3D/OG/5fold/flt_o/binary/ae_x'},
            # #    '3d_flt-x_ae-x_Res-10' : {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_10_3D/OG/5fold/flt_x/binary/ae_x'},
            # #    '3d_flt-x_ae-x_Res-18' : {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_18_3D/OG/5fold/flt_x/binary/ae_x'},
            # #    '3d_flt-x_ae-x_Res-50' : {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_50_3D/OG/5fold/flt_x/binary/ae_x'},
               '3d_flt-o_ae-x_Res-10' : {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_10_3D/OG/5fold/flt_o/binary/ae_x'},
               '3d_flt-o_ae-x_Res-18' : {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_18_3D/OG/5fold/flt_o/binary/ae_x'},
               '3d_flt-o_ae-x_Res-50' : {'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/Res_50_3D/OG/5fold/flt_o/binary/ae_x'},
            #    '3d_flt-o_ae-x_CNN' :{'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/CV5FC2_3D/OG/5fold/binary/flt_o/ae_x'},
            #    '3d_flt-o_ae-o_CNN' :{'roc_dir':'./Data/output/roc_plot/AMD_CSC_DR_RVO/CV5FC2_3D/OG/5fold/binary/flt_o/ae_o'},
               }
    

    total_tpr = {}
    total_fpr = {}
    total_auc = {}
    total_out = {}
    mean = {}
    std = {}
    for key, config in configs.items():
        model_name = get_plot_name(key)
        total_out.update({model_name:{'valid':{'gt':[], 'pd':[]}, 'test':{'gt':[], 'pd':[]}}})
        mean_tpr, mean_fpr, mean_auc, total_out[model_name], m, s = kfold_roc_plot(key, config, 'valid', total_out[model_name])
        mean[model_name]={'valid':m}
        std[model_name]={'valid':s}
        total_tpr[key] = mean_tpr
        total_fpr[key] = mean_fpr
        total_auc[key] = mean_auc
        print(f'{key} roc saved.')

    total_mean_plot(total_tpr, total_fpr, total_auc, configs)
    total_out = total_test_plot(configs, total_out)
    bar_plot(total_out, mean, std)
    
    

main()