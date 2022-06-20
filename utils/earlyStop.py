import numpy as np

class EarlyStopping():
    def __init__(self, key={'name':'ACC', 'mode':'max'}, tolerance=0., patience=10):
        '''
        mode(str) : min/max - check mode
        tolerance(float) : 0~1 - tolerance rate
        patience(int) : patience count for early stopping
        '''
        self.name = key['name']
        self.mode = key['mode']
        self.tolerance = tolerance # float between 0~1
        self.patience = patience
        self.best = -np.inf if self.mode=='max' else np.inf
        self.bad_counts = 0

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):

        if metrics is None:
            return False
        else:
            current = metrics[self.name]

        if np.isnan(current):
            return True

        if self.is_better(current, self.best):
            self.bad_counts = 0
            print(f'{self.name} - Best/Curr :{self.best:.5f}/{current:.5f} - Best')
            self.best = current
        else:
            self.bad_counts += 1
            print(f'{self.name} - Best/Curr :{self.best:.5f}/{current:.5f} - count[{self.bad_counts}]')

        if self.bad_counts >= self.patience:
            print("Early Stopped.")
            return True

        return False

    def is_better(self, current, best):
        if self.mode not in {'min', 'max'}:
            raise ValueError("mode " + self.mode + " is unknown!")

        if self.mode == 'min':
            return current < (best if best==np.inf else best - best*self.tolerance)
        else:
            return current > (best if best==-np.inf else best + best*self.tolerance)
