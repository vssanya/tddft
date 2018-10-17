class ProgressBar():
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)

    Example usage
    """

    def __init__(self, total, prefix='Progress', suffix='Complete', decimals=1, length=100, fill = '█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix

        self.decimals = decimals
        self.length = length
        self.fill = fill

        self.percentFormat = "{0:." + str(decimals) + "f}"

    def print(self, iteration):
        percent = self.percentFormat.format(100 * (iteration / float(self.total)))
        filledLength = int(self.length * iteration // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end = '\r')
        # Print New Line on Complete
        if iteration == self.total:
            print()


from time import sleep

items = list(range(0, 57))
bar = ProgressBar(len(items))
for i, item in enumerate(items):
    sleep(0.1)
    bar.print(i)
