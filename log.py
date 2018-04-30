# -*- coding:UTF8 -*-
#
 
import os
import logging
 
class Logger(object):
    '''
    @summary:日志处理对象,对logging的封装
    '''
    def __init__(self,logfile):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(logfile)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s[line:%(lineno)d]  - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        self.logger.addHandler(handler)
        self.logger.addHandler(console)
    def info(self,message):
        self.logger.info(message) 
    def error(self,message):
        self.logger.error(message)