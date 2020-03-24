# xxx revise it on 01/09, add parent
# Add attn_size in input_data, data_producer; add change_yT for indicating whether to remove the location of unk(just label it as unk)
# refactor the code of contructing the long line (def padding_and_concat)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
import time
from collections import Counter, defaultdict
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# train_sibling_file = '/home1/weibl/mt_model/data/py_train_sibling.txt'
# valid_sibling_file = '/home1/weibl/mt_model/data/py_valid_sibling.txt'
train_path_file = '/data/liufang/mt/PY_data/py_train_par_path.txt'
valid_path_file = '/data/liufang/mt/PY_data/py_valid_par_path.txt'


def input_data(N_filename, T_filename):
    start_time = time.time()
    with open(N_filename, 'rb') as f:
        print("reading data from ", N_filename)
        save = pickle.load(f)
        train_dataN = save['trainData']
        test_dataN = save['testData']
        train_dataP = save['trainParent']
        test_dataP = save['testParent']
        vocab_sizeN = save['vocab_size']
        print('the vocab_sizeN is %d (not including the eof)' % vocab_sizeN)
        print('the number of training data is %d' % (len(train_dataN)))
        print('the number of test data is %d\n' % (len(test_dataN)))

    with open(T_filename, 'rb') as f:
        print("reading data from ", T_filename)
        save = pickle.load(f)
        train_dataT = save['trainData']
        test_dataT = save['testData']
        vocab_sizeT = save['vocab_size']
        attn_size = save['attn_size']
        print('the vocab_sizeT is %d (not including the unk and eof)' % vocab_sizeT)
        print('the attn_size is %d' % attn_size)
        print('the number of training data is %d' % (len(train_dataT)))
        print('the number of test data is %d' % (len(test_dataT)))
        print('Finish reading data and take %.2f\n' % (time.time() - start_time))


    return train_dataN, test_dataN, vocab_sizeN, train_dataT, test_dataT, vocab_sizeT, attn_size, train_dataP, test_dataP


def raw_data_producer(raw_data, batch_size, num_steps, vocab_size, change_yT=False, name=None, verbose=False):

    with tf.name_scope(name, "DataProducer", [raw_data, batch_size, num_steps, vocab_size]):
        (raw_dataN, raw_dataT, raw_dataP) = raw_data
        assert len(raw_dataN) == len(raw_dataT)

        (vocab_sizeN, vocab_sizeT) = vocab_size
        eof_N_id = vocab_sizeN - 1
        eof_T_id = vocab_sizeT - 1
        unk_id = vocab_sizeT - 2

        def padding_and_concat(data, width, pad_id, path_len = None):
            # the size of data: a list of list. This function will pad the data according to width
            long_line = list()
            for line in data:
                pad_len = width - (len(line) % width)
                if path_len:
                    new_line = line + [[pad_id] * path_len] * pad_len # [seq_len, path_len]
                else:
                    new_line = line + [pad_id] * pad_len
                #new_line = line + [pad_id] * pad_len
                assert len(new_line) % width == 0
                long_line += new_line
            return long_line

        pad_start = time.time()
        long_lineN = padding_and_concat(raw_dataN, num_steps, pad_id=eof_N_id)
        long_lineT = padding_and_concat(raw_dataT, num_steps, pad_id=eof_T_id)
        long_lineP = padding_and_concat(raw_dataP, num_steps, pad_id=1)

        #assert len(long_lineN) == len(long_lineT) == len(long_linePath)
        print('Pading three long lines and take %.2fs' % (time.time() - pad_start))

        # print statistics for long_lineT
        if verbose:
            print('Start counting the statistics of T!!')
            verbose_start = time.time()
            cnt_T = Counter(long_lineT)
            long_lineT_len = len(long_lineT)
            empty_cnt = cnt_T[0]
            unk_cnt = cnt_T[unk_id]
            eof_cnt = cnt_T[eof_T_id]
            l_cnt = sum(np.array(long_lineT) > eof_T_id)
            w_cnt = long_lineT_len - empty_cnt - unk_cnt - eof_cnt - l_cnt
            print('long_lineT_len: %d, empty: %.4f, unk: %.4f, location: %.4f, eof: %.4f, word (except Empty): %.4f' %
                  (long_lineT_len, float(empty_cnt) / long_lineT_len, float(unk_cnt) / long_lineT_len,
                   float(l_cnt) / long_lineT_len, float(eof_cnt) / long_lineT_len, float(w_cnt) / long_lineT_len))
            print('the most common 5 of cnt_T', cnt_T.most_common(5))
            print('print verbose information and take %.2fs\n' % (time.time() - verbose_start))

        temp_len = len(long_lineN)
        n = temp_len // (batch_size * num_steps)
        long_lineN_truncated = np.array(long_lineN[0: n * (batch_size * num_steps)])
        long_lineP_truncated = np.array(long_lineP[0: n * (batch_size * num_steps)])
        long_lineT_truncated_x = np.array(long_lineT[0: n * (batch_size * num_steps)])
        long_lineT_truncated_y = np.array(long_lineT[0: n * (batch_size * num_steps)])
        #long_lineS_truncated = np.array(long_lineS[0: n * (batch_size * num_steps)])

        # long_lineP_truncated[long_lineP_truncated > attn_size] = attn_size  #if the parent location is too far
        long_lineP_truncated = [long_lineN_truncated[i - j] for i, j in
                                enumerate(long_lineP_truncated)]  # only store parent N

        location_index = long_lineT_truncated_x > eof_T_id
        long_lineT_truncated_x[location_index] = unk_id
        if change_yT:
            long_lineT_truncated_y[location_index] = unk_id

        data_len = len(long_lineN_truncated)
        batch_len = data_len // batch_size

        epoch_size = (batch_len - 1) // num_steps  # how many batches to complete a epoch
        assert epoch_size > 0

        dataN = np.reshape(long_lineN_truncated[0: batch_size * batch_len], [batch_size, batch_len]).astype(np.int32)
        #dataS = np.reshape(long_lineS_truncated[0: batch_size * batch_len], [batch_size, batch_len]).astype(np.int32)
        dataP = np.reshape(long_lineP_truncated[0: batch_size * batch_len], [batch_size, batch_len]).astype(np.int32)
        dataT_x = np.reshape(long_lineT_truncated_x[0: batch_size * batch_len], [batch_size, batch_len]).astype(
            np.int32)
        dataT_y = np.reshape(long_lineT_truncated_y[0: batch_size * batch_len], [batch_size, batch_len]).astype(
            np.int32)

        return dataN, dataP, dataT_x, dataT_y


def real_data_producer(data, batch_size, num_steps, vocab_size):
    (dataN, dataP, dataT_x, dataT_y) = data
    epoch_size = (dataN.shape[0] * dataN.shape[1] // batch_size - 1) // num_steps
    print('total epochs {}'.format(epoch_size))
    (vocab_sizeN, vocab_sizeT) = vocab_size
    eof_N_id = vocab_sizeN - 1
    eof_T_id = vocab_sizeT - 1
    unk_id = vocab_sizeT - 2

    for i in range(epoch_size):
        xN = dataN[:, i * num_steps:(i + 1) * num_steps]
        yN = dataN[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        xT = dataT_x[:, i * num_steps:(i + 1) * num_steps]
        yT = dataT_y[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        xP = dataP[:, i * num_steps:(i + 1) * num_steps]
        eof_indicator = np.equal(xN[:, num_steps - 1], np.ones(batch_size).astype(np.int32) * eof_N_id)
        yield xN, yN, xT, yT, epoch_size, eof_indicator, xP
'''
if __name__ == '__main__':
    N_filename = '../pickle_data/PY_non_terminal.pickle'
    T_filename = '../pickle_data/PY_terminal_50k_whole.pickle'

    train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size, train_dataP, valid_dataP, train_dataS, valid_dataS \
        = input_data(N_filename, T_filename)
    train_data = (train_dataN, train_dataT, train_dataP, train_dataS)
    valid_data = (valid_dataN, valid_dataT, valid_dataP, valid_dataS)
    vocab_size = (vocab_sizeN + 1, vocab_sizeT + 2)  # N is [w, eof], T is [w, unk, eof]
    # print(py_NT_id_to_word)
    dataN, dataP, dataT_x, dataT_y, dataS = \
        raw_data_producer(train_data, batch_size=10, num_steps=5, vocab_size=vocab_size,
                      change_yT=False, name='train', verbose=False)
    data = (dataN, dataP, dataT_x, dataT_y, dataS)
    data_loader=real_data_producer(data, 10, 5, vocab_size)

    xN, yN, xT, yT, epoch_size, eof_indicator, xP, xS, yS = next(data_loader)
'''
