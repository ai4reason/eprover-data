import tensorflow as tf
import random as rnd
import numpy as np
from tensorflow.contrib.layers import fully_connected
from stop_watch import StopWatch, print_times
from tf_helpers import *
import fcoplib as cop
import os

import debug_node
from debug_node import tf_debug
from parallel_environment import GameState
from graph_placeholder import GraphPlaceholder
from graph_conv import graph_start, graph_conv
from segments import Segments, SegmentsPH
from graph_data import GraphData

class NetworkConfig:
    def __init__(self):
        self.threads = 1
        self.start_shape = (4,1,4)
        self.next_shape = (32,64,32)
        #self.next_shape = (11,12,13)
        #self.res_blocks = 3
        #self.layers = 3
        self.res_blocks = 1
        self.layers = 5
        self.hidden = 128
        self.symbol_loss_ratio = 0
        self.balance_loss = True

    def __str__(self):
        return "start {}, next {}, last hidden {}, {} x {} layers, symbol {}, bal_loss {}".format(
            self.start_shape, self.next_shape, self.hidden,
            self.layers, self.res_blocks, self.symbol_loss_ratio, self.balance_loss,
        )

class Network:
    def __init__(self, total_symbols = None, config = None):
        if config is None: config = NetworkConfig()
        print(config)
        self.config = config

        graph = tf.Graph()
        graph.seed = 43

        self.session = tf.Session(graph = graph,
                                  config=tf.ConfigProto(inter_op_parallelism_threads=config.threads,
                                                        intra_op_parallelism_threads=config.threads))

        with self.session.graph.as_default():

            self.structure = GraphPlaceholder()

            x = graph_start(self.structure, config.start_shape)
            last_x = None
            for _ in range(config.res_blocks):
                for n in range(config.layers):
                    x = graph_conv(x, self.structure,
                                   output_dims = config.next_shape, use_layer_norm = False)
                    #x = tuple(map(layer_norm, x))
                if last_x is not None:
                    x = [cx+lx for cx, lx in zip(x, last_x)]
                last_x = x

            nodes, symbols, clauses = x

            self.total_symbols = total_symbols
            if total_symbols is not None:
                self.symbol_num = tf.shape(symbols)[0]
                symbol_logits = tf_linear(symbols, total_symbols)

                symbol_labels = tf.placeholder(tf.int64, [None])
                self.symbol_labels = symbol_labels

                self.symbol_loss = tf.losses.sparse_softmax_cross_entropy(
                    symbol_labels, symbol_logits
                )
                symbol_predictions = tf.argmax(symbol_logits, 1)
                self.symbol_predictions = symbol_predictions
                self.symbol_accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(symbol_labels, symbol_predictions),
                        tf.float32,
                    )
                )
            else:
                self.symbol_num = tf.constant(1)
                self.symbol_loss = tf.constant(0)
                self.symbol_accuracy = tf.constant(0)

            prob_segments = SegmentsPH(nonzero = True)
            self.prob_segments = prob_segments

            theorems = Segments(prob_segments.data).collapse(clauses)
            conjectures = prob_segments.gather(theorems, 0)
            mask = 1 - tf.scatter_nd(
                tf.expand_dims(prob_segments.start_indices_nonzero, 1),
                tf.ones(tf.reshape(prob_segments.nonzero_num, [1]), dtype=tf.int32),
                [prob_segments.data_len],
            )
            prem_segments, premises = prob_segments.mask_data(
                theorems, mask,
                nonzero = True
            )

            network_outputs = tf.concat(
                [premises, prem_segments.fill(conjectures)],
                axis = 1
            )
            hidden = fully_connected(network_outputs, config.hidden)
            premsel_logits = tf_linear_sq(hidden)

            premsel_labels = tf.placeholder(tf.int32, [None])
            self.premsel_labels = premsel_labels

            pos_mask = tf.cast(premsel_labels, tf.bool)
            neg_mask = tf.logical_not(pos_mask)

            premsel_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.cast(premsel_labels, tf.float32),
                logits = premsel_logits,
            )
            if config.balance_loss:
                loss_on_true = tf.boolean_mask(premsel_loss, pos_mask)
                loss_on_false = tf.boolean_mask(premsel_loss, neg_mask)
                self.premsel_loss = (mean_or_zero(loss_on_true) + mean_or_zero(loss_on_false))/2
            else: self.premsel_loss = tf.reduce_mean(premsel_loss)

            if config.symbol_loss_ratio == 0:
                loss = self.premsel_loss
            else:
                loss = self.premsel_loss + config.symbol_loss_ratio * self.symbol_loss

            optimizer = tf.train.AdamOptimizer()
            self.training = optimizer.minimize(loss)

            premsel_predictions = tf.cast(tf.greater(premsel_logits, 0), tf.int32)
            self.premsel_predictions = premsel_predictions
            self.premsel_num = tf.size(premsel_predictions)
            #self.premsel_accuracy = tf.reduce_mean(
            #    tf.cast(
            #        tf.equal(premsel_labels, premsel_predictions),
            #        tf.float32,
            #    )
            #)
            predictions_f = tf.cast(premsel_predictions, tf.float32)
            predictions_on_true = tf.boolean_mask(predictions_f, pos_mask)
            predictions_on_false = tf.boolean_mask(predictions_f, tf.logical_not(pos_mask))
            self.premsel_tpr = mean_or_zero(predictions_on_true)
            self.premsel_tnr = mean_or_zero(1-predictions_on_false)

            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

        self.session.graph.finalize()

    def feed(self, data, use_labels, non_destructive = True):
        graph_data, lens_labels_symbols = zip(*data)
        d = self.structure.feed(graph_data, non_destructive)
        prob_lens, labels, symbols = zip(*lens_labels_symbols)
        self.prob_segments.feed(d, prob_lens)
        if use_labels:
            d[self.premsel_labels] = np.concatenate(labels)
            if self.total_symbols is not None:
                d[self.symbol_labels] = np.concatenate(symbols)
        return d

    def predict(self, data):
        with StopWatch("data preparation"):
            d = self.feed(data, use_labels = False)
        with StopWatch("network"):
            return self.session.run(self.premsel_predictions, self.symbol_predictions, d)

    def get_loss(self, data):

        with StopWatch("data preparation"):
            d = self.feed(data, use_labels = True)
        with StopWatch("network"):
            return self.session.run(
                ((self.premsel_loss, self.premsel_tpr, self.premsel_tnr),
                 (self.symbol_loss, self.symbol_accuracy),
                 self.premsel_num, self.symbol_num), d)

    def train(self, data):

        with StopWatch("data preparation"):
            d = self.feed(data, use_labels = True)
        with StopWatch("network"):
            return self.session.run(
                (self.training,
                 (self.premsel_loss, self.premsel_tpr, self.premsel_tnr),
                 (self.symbol_loss, self.symbol_accuracy),
                 self.premsel_num, self.symbol_num), d)[1:]

    def debug(self, data, labels = None):

        d = self.feed(data, use_labels = True)
        debug_node.tf_debug_print(self.session.run(
            debug_node.debug_nodes, d
        ))

    def save(self, path, step = None):
        self.saver.save(self.session, path, global_step = step, write_meta_graph=False, write_state=False)

    def load(self, path):
        self.saver.restore(self.session, path)

def file_lines(fname):
    with open(fname) as f:
        return [ line.strip() for line in f ]

def load_data(datadir, train_datadir = None):
    if train_datadir is None:
      fnames = os.listdir(datadir)
      rnd.shuffle(fnames)
      test_num = len(fnames) // 10
      test_fnames = fnames[:test_num]
      train_fnames = fnames[test_num:]
      train_datadir = datadir
      test_datadir = datadir
    else:
      test_datadir = datadir
      train_fnames = os.listdir(train_datadir)
      test_fnames = os.listdir(test_datadir)
    #if isinstance(test_fnames, str): test_fnames = file_lines(test_fnames)
    #if isinstance(train_fnames, str): train_fnames = file_lines(train_fnames)

    test_data = []
    for fname in test_fnames:
        data, lens_labels_symbols = cop.load_premsel(os.path.join(test_datadir, fname))
        test_data.append((GraphData(data), lens_labels_symbols))

    train_data = []
    for fname in train_fnames:
        data, lens_labels_symbols = cop.load_premsel(os.path.join(train_datadir, fname))
        train_data.append((GraphData(data), lens_labels_symbols))

    return test_data, train_data

def enumerate_symbols(*data_lists):
    def truncate_skolem_single(symbol):
        if symbol.startswith("'skolem"): return "skolem"
        if symbol.startswith("'def"): return "def"
        return symbol
    def truncate_skolem(symbols):
        return map(truncate_skolem_single, symbols)
    symbol_set = set()
    for data in data_lists:
        for graph_data, (lens, labels, funcs_rels) in data:
            funcs, rels = funcs_rels
            symbol_set.update(truncate_skolem(funcs+rels))

    symbol_to_num = dict(
        (symbol, i)
        for i,symbol in enumerate(sorted(symbol_set))
    )

    res = [symbol_to_num]
    for data in data_lists:
        res_data = []
        for graph_data, (lens, labels, (funcs, rels)) in data:
            symbols = [
                symbol_to_num[symbol]
                for symbol in truncate_skolem(funcs+rels)
            ]
            res_data.append((graph_data, (lens, labels, symbols)))
        res.append(res_data)
    return res

if __name__ == "__main__":
    import traceback_utils
    import sys

    # hide entrails of Tensorflow in error messages
    sys.excepthook = traceback_utils.shadow('/home/mirek/.local/')

    with StopWatch("loading data"):

        print("Loading data...")
        sys.stdout.flush()
        #test_data, train_data = load_data("deepmath/nndata2")
        #test_data, train_data = load_data("bartosz/nndata/test", "bartosz/nndata/train")
        test_data, train_data = load_data("enigma-2019-10/all-mzr02/test_sng", "enigma-2019-10/all-mzr02/train_sng")
        print("Enumerate symbols...")
        symbol_to_num, test_data, train_data = enumerate_symbols(test_data, train_data)
        #total_symbols = len(symbol_to_num)
        #print("{} symbols".format(total_symbols))
        print("Constructing network...")
        sys.stdout.flush()

    with StopWatch("network construction"):

        network = Network()
        #network.load("weights/premsel_bartosz_bal_29")
        #network.debug(test_data)

    epochs = 20
    premsel_accum = [1.0, 0.0, 0.0]
    symbol_accum = [1.0, 0.0]
    def update_accum(accum, current):
        for i,(acc,cur) in enumerate(zip(accum, current)):
            accum[i] = np.interp(0.1, [0, 1], [acc, cur])
    def stats_str(stats):
        if len(stats) == 2:
            return "loss {:.4f}, acc {:.4f}".format(*stats)
        else: return "loss {:.4f}, acc {:.4f} ({:.4f} / {:.4f})".format(stats[0], (stats[1]+stats[2])/2, stats[1], stats[2])

    #batch_size = 50
    batch_size = 200
    for epoch_i in range(0, epochs):

        with StopWatch("training"):

            rnd.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):

                #if i == 1: network.debug(test_data[:50])
                if (i//batch_size)%100 == 0:
                    if network.config.symbol_loss_ratio == 0: symbols_str = ""
                    else: symbols_str = "; Symbols "+stats_str(symbol_accum)
                    print('Training {}: {} / {}: Premsel {}{}'.format(
                        epoch_i, i, len(train_data),
                        stats_str(premsel_accum), symbols_str,
                    ))
                    sys.stdout.flush()

                batch = train_data[i : i+batch_size]
                premsel_cur, symbol_cur, _, _ =  network.train(batch)

                update_accum(premsel_accum, premsel_cur)
                update_accum(symbol_accum, symbol_cur)

            network.save("weights/premsel_enigma_sng_all_{}".format(epoch_i))
            #network.debug(test_data[:50])

        with StopWatch("evaluation"):

            premsel_stats = []
            premsel_nums = []
            symbol_stats = []
            symbol_nums = []
            for i in range(0, len(test_data), batch_size):

                batch = test_data[i : i+batch_size]
                premsel_cur, symbol_cur, premsel_num, symbol_num =  network.get_loss(batch)

                premsel_stats.append(np.array(premsel_cur))
                premsel_nums.append(premsel_num)
                symbol_stats.append(np.array(symbol_cur))
                symbol_nums.append(symbol_num)

            premsel_stats = np.average(premsel_stats, weights = premsel_nums, axis = 0)
            symbol_stats = np.average(symbol_stats, weights = symbol_nums, axis = 0)
            print()
            if network.config.symbol_loss_ratio == 0: symbols_str = ""
            else: symbols_str = "; Symbols "+stats_str(symbol_stats)
            print('Testing {}: Premsel {}{}'.format(
                epoch_i,
                stats_str(premsel_stats), symbols_str,
            ))
            print()
            sys.stdout.flush()

    print_times()
