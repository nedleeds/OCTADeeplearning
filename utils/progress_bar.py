def progress_bar(iterable, prefix = '', suffix = '', decimals = 1, 
                 length = 100, fill = '█', printEnd = "\r", 
                 label = None, cnt_mode = False, idx = False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        cnt_mode    - Optional  : return count of iteration.
        idx         - Optional  : return the index of iteration.
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration, item=None):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        if cnt_mode:
            print(f'\r{prefix} |{bar}| {suffix} {iteration}', end = printEnd, flush=True)
        elif (item is not None):
            if (label is not None):
                print(f'\rLoading({percent}%) : |{bar}| [{item:5d}]-{label[item]:<6}', end = printEnd, flush=True)
            else:
                print(f'\rLoading({percent}%) : |{bar}| [{item[0]:5d}]-{item[1]:<6}', end = printEnd, flush=True)
        else:
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd, flush=True)
            
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item if idx is True else i, item
        if label is not None or len(item)>1:
            printProgressBar(i + 1, item)
        else:
            printProgressBar(i + 1)
    # Print New Line on Complete
    print()
    
    
'''
ex)
import time

# A List of Items
items = list(range(0, 57))

# A Nicer, Single-Call Usage
for item in progressBar(items, prefix = 'Progress:', suffix = 'Complete', length = 50):
    # Do stuff...
    time.sleep(0.1)
'''