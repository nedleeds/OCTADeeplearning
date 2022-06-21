def progress_bar(iterable, prefix = '', suffix = '', decimals = 1, 
                 length = 100, fill = '█', printEnd = "\r", cnt_mode = False):
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
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        if cnt_mode:
            print(f'\r{prefix} |{bar}| {suffix} {iteration}', end = printEnd)
        else:
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
            
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
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