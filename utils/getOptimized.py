from typing import Any

class Parameters():
    def __init__(self, parameters) -> None:
        self._hyper_parameters = parameters
        self._checkValue = {}
        self._mode = ''        
        self._fold = 0
        self._key = ''

    def getCheckValue(self):
        return self._checkValue
    def getHyperParameters(self):
        return self._hyper_parameters
    def setCheckValue(self, check):
        self._checkValue=check
    def selectParameter(self):
        pass
    def isOptimized(self, checkValues):
        pass
    def isSelected(self, current_parameter):
        pass
    def saveParameter(self,file_path):
        pass
    def __str__(self):
        return list(self._hyper_parameters.keys())

# class GridSearch(Parameters):
#     def __init__(self, parameters) -> None:
#         super().__init__(parameters)


class RandomSearch(Parameters):
    def __init__(self, parameters, fold_num, path) -> None:
        super().__init__(parameters)
        self._current_parameter = {}
        self._best_parameter = {f'fold{idx}':{} for idx in range(1, fold_num+1)}
        self._selected = []
        self._function_call_num = 0
        self._fold_num = fold_num
        self._value = {'init':0., 'last':0.}
        self._curr_parms_path=f'{path}/current_parameter.txt'
        self._best_parms_path=f'{path}/best_parameter.txt'
        self._key = ''

    def setKey(self, key='MeanGradient', value=0., epoch=None):
        if epoch == None:
            self._key = {'name':key, 'value':value}
        else:
            self._key = {'name':key, 'value':value, 'epoch':epoch}

    def getKey(self):
        return self._key

    def isSelected(self, current_parameter):
        if current_parameter in self._selected:
            return False
        else:
            self._selected.append(current_parameter)
            return True

    def selectParameter(self, fold=None):
        import random
        hyperParameters = self.getHyperParameters()
        current_parameter = {}
        isNew = False

        while not isNew:
            for name, value in hyperParameters.items():
                current_parameter[name] = random.choice(value)
            param_values = list(current_parameter.values())
            isNew = self.isSelected(param_values)

        self._current_parameter = current_parameter
        self.saveParameter(file_path=self._curr_parms_path, fold_idx=fold)
        
        return current_parameter

    def saveParameter(self, file_path, fold_idx, best=False):

        self._save_path = self._best_parms_path if best else self._curr_parms_path
        
        hyper_params = list(self._hyper_parameters.keys())
        if self._function_call_num==0 and fold_idx==1:
            self.makeSaveFile(hyper_params)

        p = self._best_parameter if best else self._current_parameter
        with open(self._save_path,'a') as f:
            if best:
                f.write(f"{fold_idx:<4d}\t{p[f'fold{fold_idx}']['Batch']:<5d}\t{p[f'fold{fold_idx}']['LearningRate']:<12.0E}\t{p[f'fold{fold_idx}']['Gamma']:<5.2f}\t")
            else:
                f.write(f"{fold_idx:<4d}\t{p['Batch']:<5d}\t{p['LearningRate']:<12.0E}\t{p['Gamma']:<5.2f}\t")
        self._function_call_num += 1

    def makeSaveFile(self, hyper_params):
        with open(self._curr_parms_path,'w') as f:
                f.write(f"Fold\t"+('\t').join(hyper_params)+f"\t{self._key['name']}\n")
        with open(self._best_parms_path, 'w') as f:
                f.write(f"Fold\t"+('\t').join(hyper_params)+f"\t{self._key['name']}\n")

    def saveResult(self, result):
        with open(self._save_path,'a') as f:
            f.write(f"{result:.10f}\n")

    def isOptimized(self, value=0., epoch=None, fold_idx=None):
        keys = self.getKey()
        keyName = keys['name']
        keyEpoch = keys['epoch']
        isOptimized = False

        if keyName.lower() =='meangradient':
            if epoch == keyEpoch:
                self._value['last']=value
                isOptimized = self.checkGradient(epoch, fold_idx)
                return isOptimized
            elif epoch == 1:
                self._value['init']=value
            else: 
                pass
        else:
            pass
        return isOptimized
    
    def checkGradient(self, epoch, fold_idx):
        self._mean_gradient = (self._value['init']-self._value['last'])/(0-epoch)
        self.saveResult(self._mean_gradient)
        # if (0.615-0.45)/(0-30) > mean_gradient >= (0.73-0.45)/(0-30): 
        # if (0.65-0.45)/(0-epoch) > self._mean_gradient: 
        if -(0.00007)>self._mean_gradient:
            self._best_parameter[f'fold{fold_idx}'] = self._current_parameter
            self.saveParameter(self._best_parms_path, fold_idx, best=True)
            self.saveResult(self._mean_gradient)
        # if self._mean_gradient<0:
            return True
        else : 
            print("\nThis model is not good.\n")
            return False

    def getBestParams(self):
        return self._best_parameter

    def getMeanGradient(self):
        return self._mean_gradient
