import sys


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t)/60
        hr = t//60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode=='sec':
        t = int(t)
        min = t//60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
       lr += [param_group['lr']]

    assert(len(lr) == 1)  # we support only one param_group
    lr = lr[0]

    return lr
