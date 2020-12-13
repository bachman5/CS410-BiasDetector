from SVM_Model.work2vec.Utils import Utils
from SVM_Model.work2vec.config import *


if __name__ == '__main__':
    #Utils.prepare_train_test_and_corpus(DATA_DIR, TRAIN_FF, TEST_FF)
    Utils.read_label_data(LABELED_DATA_DIR, TRAIN_FF, TEST_FF)

# STOP HERE! READ ME!
# STOP HERE! READ ME!
# Now: proceed to create the .magnitude file manually, then come back here and resume
"""
Now copy content of either CORPUS_TRAIN_FF or CORPUS_ALL_FF to ../glove_genvecs/text8
Then follow instruction on http://github.com/stanfordnlp/glove to generate word embedded vectors db as a .txt file
Then follow https://github.com/plasticityai/magnitude#file-format-and-converter to generate .magnitude file from .txt
"""