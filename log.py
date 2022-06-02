# -*- coding:utf8 -*-

import logging


class Logger():
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self, filename, level='debug'):
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations.get(level))
        handler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(lineno)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
