import random
from os import listdir
from os.path import isfile, join

import pandas as pd
from pymagnitude import MagnitudeUtils
from SVM_Model.work2vec.Tokenizer import Tokenizer as Toki
from SVM_Model.work2vec.config import *


class Utils(object):
    def __init__(self):
        pass

    # Exmaple train file: http://magnitude.plasticity.ai/data/atis/atis-intent-train.txt
    # Exmaple test file : http://magnitude.plasticity.ai/data/atis/atis-intent-test.txt
    @staticmethod
    def format_for_glove(text: str, target: str):
        #return "BOS {} EOS {}".format(text, target)
        return "{} {}".format(text, target)

    @staticmethod
    def read_label_data(data_dir: str, train_ff: str, test_ff: str):
        ffs = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('.csv')]
        target_to_data = {}
        for ff in ffs:
            ff_path = join(data_dir, ff)
            df = pd.read_csv(ff_path)

            # retrieve data as a list of tuple ($Title, $Bias)
            items = df[['Title', 'Bias']].to_records(index=False)

            for item in items:
                headline = item[0]
                target = item[1].strip().lower()
                line_to_write = Utils.format_for_glove(headline, target)

                if target not in target_to_data:
                    target_to_data[target] = []
                selected_group = target_to_data[target]
                selected_group.append(line_to_write)

        group_names = sorted(target_to_data.keys())

        # shuffle arrays to guarantee unbias training
        for name in group_names:
            random.shuffle(target_to_data[name])
            print("{} = {}".format(name, len(target_to_data[name])))

        # the train/test split ratio has to applied separate to both lefts & rights, so that in either train or
        # test set, our democrat count / republican count == same
        cutoffs = [int(len(target_to_data[name])*TRAIN_RATIO) for name in group_names]
        trains = []
        tests = []
        for name, cof in zip(group_names, cutoffs):
            gg = target_to_data[name]
            trains += gg[:cof]
            tests += gg[cof:]
        random.shuffle(trains)
        random.shuffle(tests)
        print("trains =", len(trains))
        print("tests =", len(tests))

        # now have 2 arrays trains & tests, we proceed to write them to respective files
        with open(train_ff, 'w+') as ftrain:
            for ll in trains:
                ftrain.write(ll + "\n")
        with open(test_ff, 'w+') as ftest:
            for ll in tests:
                ftest.write(ll + "\n")

        # build corpus
        # global thing to do: write to corpus txt for later generation of .magnitude database (word -> embedded vector)
        with open(CORPUS_TRAIN_FF, 'w+') as f_corpus_train, open(CORPUS_ALL_FF, 'w+') as f_corpus_all:
            for ll in trains:
                # do not include the target word
                sent = ll.rsplit(' ', 1)[0]
                sent = Toki.normalize_text(sent)
                f_corpus_train.write(sent + "\n")
            for ll in trains + tests:
                # do not include the target word
                sent = ll.rsplit(' ', 1)[0]
                sent = Toki.normalize_text(sent)
                f_corpus_all.write(sent + "\n")
        pass

    @staticmethod
    def read_train_test_data(train_ff: str, test_ff: str):
        with open(train_ff, 'rb') as ff:
            trains = [line.decode('utf-8') for line in ff.readlines()]
        with open(test_ff, 'rb') as ff:
            tests = [line.decode('utf-8') for line in ff.readlines()]

        """
            etree_w2v = Pipeline([
                ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                ("extra trees", ExtraTreesClassifier(n_estimators=200))])
            """
        add_label, label_to_int, int_to_label = MagnitudeUtils.class_encoding()

        def get_sentence(line: str):
            return line.rsplit(' ', 1)[0]

        def get_target(line: str):
            return line.rsplit(' ', 1)[-1].strip().lower()

        X_train = [get_sentence(line) for line in trains]
        y_train = [add_label(get_target(line)) for line in trains]

        X_test = [get_sentence(line) for line in tests]
        y_test = [add_label(get_target(line)) for line in tests]

        return X_train, y_train, X_test, y_test, label_to_int, int_to_label

    #Build corpus function for category data (news, L/R/trump)
    @staticmethod
    def build_corpus_and_data_all(dir: str):
        files = listdir(dir)
        with open(CORPUS_ALL_FF, 'w+') as fcorpus, open(ALL_FF, 'w+') as fAll:
            for ff in files:
                dd = join(dir, ff)
                label = dd.replace('.csv', '').rsplit('_', 1)[1].lower()
                titles = pd.read_csv(dd)[["title"]].to_records(index=False)  # return a tuble
                print(dd)
                if label == 'all':
                    for tt in titles:
                        try:
                            if tt[0].strip():
                                title = tt[0]
                                fcorpus.write(Toki.normalize_text(title) + "\n")
                        except:
                            continue
                else:
                    for tt in titles:
                        try:
                            if tt[0].strip():
                                title = tt[0].strip() + ' ' + label
                                fAll.write(title + "\n")
                        except:
                            continue

    # Create train/Test file function for category data (news, L/R/trump)
    @staticmethod
    def create_train_test_files(train_ff, test_ff, file_all):
        label_to_title = dict()
        with open(file_all, 'rb') as ff:
            lines = [line.decode('utf-8') for line in ff.readlines()]
        # print(lines[0])

        for line in lines:
            if line.strip():
                whole_line = line.strip()
                #print("Checking:" + whole_line + "////")
                lable = whole_line.rsplit(' ', 1)[1]

                if lable not in label_to_title:
                    label_to_title[lable] = []
                label_to_title[lable].append(whole_line)

        labels_names = sorted(label_to_title.keys())
        # separate train data & test data using ratio
        cutoffs = [int(len(label_to_title[label]) * TRAIN_RATIO) for label in sorted(label_to_title)]
        trains = []
        tests = []
        for name, cof in zip(labels_names, cutoffs):
            gg = label_to_title[name]
            trains += gg[:cof]
            tests += gg[cof:]
        random.shuffle(trains)
        random.shuffle(tests)
        for k, v in label_to_title.items():
            print("Data: {} = {}".format(k, len(v)))
        print("trains =", len(trains))
        print("tests =", len(tests))

        # now have 2 arrays trains & tests, we proceed to write them to respective files
        with open(train_ff, 'w+') as ftrain:
            for ll in trains:
                ftrain.write(ll + "\n")
        with open(test_ff, 'w+') as ftest:
            for ll in tests:
                ftest.write(ll + "\n")
