import logging.handlers

class Logger(logging.Logger):
    def __init__(self, filename = None):
        super(Logger, self).__init__(self)

        if filename is None:
            filename = 'log.txt'
        self.filename = filename

        fh = logging.handlers.TimedRotatingFileHandler(self.filename, 'D', 1, 30)
        fh.sufflx = '%Y%m%d-%H%M.log'
        fh.setLevel(logging.INFO)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] - %(filename)s [Line:%(lineno)d] - [%(levelname)s] - [thread:%(thread)s] - [process:%(process)s] - %(message)s')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        self.addHandler(fh)
        self.addHandler(sh)

if __name__ == '__main__':
    pass


