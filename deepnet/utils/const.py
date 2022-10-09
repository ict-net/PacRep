# coding: utf-8
#
# Copyright 2020 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# const file

import os


class Const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('Const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value


CONST = Const()
CONST.UNK = '<UNK>'
CONST.IGNORE = '<IGN>'
CONST.NA = '<NA>'
CONST.NOTFOUND = '<NOTFOUND>'
CONST.DELIMITER = '<_M_>'
CONST.MAX_DOC_NUM = 128
CONST.MAX_SUB_WORD_NUM = 512
CONST.EPS = 1e-9
CONST.APP_ROOT_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
