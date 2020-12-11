import os
from os.path import join

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


def __file(path: str):
    return join(WORKING_DIR, path)


#print(join(THIS_DIR, '../rawdata'))

TAR_DEMOCRAT = 'democrat'
TAR_REPUBLIC = 'republic'

TRAIN_RATIO = 0.7

LABELED_DATA_DIR = __file('../labeled_data/politics')
RAW_DATA_DIR = __file('../../data_raw')
#LABELED_DATA_DIR = __file('../labeled_data/airline')
TRAIN_FF = __file('data/train.txt')
TEST_FF = __file('data/test.txt')
ALL_FF = __file('data/all.txt')

CORPUS_TRAIN_FF = __file('data/corpus-train.txt')
CORPUS_ALL_FF = __file('data/corpus-all.txt')

CORPUS_FF = __file('corpus/vectors.magnitude')
#CORPUS_FF = __file('corpus/glove.6B.50d.magnitude')
#CORPUS_FF = __file('corpus/wiki-news-300d-1M.magnitude')

SAVED_TRANSFORMER_FF = __file('model/transformer.pkl')
SAVED_MODEL_FF = __file('model/model.pkl')


class Config(object):
    def __init__(self):
        pass
