'''
streaming client (fully-connected DNN)
'''
from log import init_logger
if __name__ == '__main__':  # only init logger for main process
  init_logger(__file__)

import client
import defaultParser
import csv
from processor import Flow_Processor
import tensorflow as tf
import math
import time
import numpy as np
import multiprocessing
from constants import *
import sys
from log import log

if __name__ == '__main__':
  print('TensorFlow version {}'.format(tf.__version__))
  if tf.__version__ != '1.14.0': print('tested with tensorflow version 1.14', file=sys.stderr)

# command line parameter example: select features from dataset
# parameter pattern: '--feature 1-[0|1|2] 2-1 3-0'

feature_selector = {
  # 0: [-1, -1, -1], # year

  #   +- floats
  #   |   +- bit pattern
  #   |   |   +- one hot
  #   |   |   |
  #   v   v   v
  0: [4, 32, -1],  # src_addr
  1: [4, 32, -1],  # dst_addr
  2: None,  # packets
  3: None,  # bytes
  4: None,  # first switched
  5: None,  # last switched
  6: [1, 16, -1],  # src_port
  7: [1, 16, -1],  # dst_port
  8: [1, 8, -1],  # tcp_flags
  9: [1, 8, -1],  # protocol
  10: None,  # export host
  11: None,  # flow seq number
  12: None,  # duration
  13: None,  # bitrate
  14: [1, 8, 250],  # src_country_code
  15: [1, -1, -1],  # src_longitude
  16: [1, -1, -1],  # src_latitude
  17: [1, 16, -1],  # src_asn
  18: [4, 32, -1],  # src_network
  19: [1, 6, -1],  # src_prefix_len
  20: [1, 12, -1],  # src_vlan
  21: [-1, 1, -1],  # src_locality
  22: [1, 8, 250],  # dst_country_code
  23: [1, -1, -1],  # dst_longitude
  24: [1, -1, -1],  # dst_latitude
  25: [1, 16, -1],  # dst_asn
  26: [4, 32, -1],  # dst_network
  27: [1, 6, -1],  # dst_prefix_len
  28: [1, 12, -1],  # dst_vlan
  29: [-1, 1, -1],  # dst_locality
  30: None,  # year
  31: [1, 4, 12],  # month
  32: [1, 5, 31],  # day
  33: [1, 5, 24],  # hour
  34: [1, 6, 60],  # minute
  35: [1, 6, 60],  # second
  }


class Stream_Client_FC_DNN(client.Stream_Client):
  ''' training class (inherited class from Stream_Client)
    1. build fully-connected DNN
    2. start thread to receive data (from inherited class)
  '''

  def __init__(self):
    client.Stream_Client.__init__(self)
    #-------------------------------------------------------------------------------------------------------- ARG PARSER
    self.block = 0
    self.FLAGS = None
    self.arg_parse()
    #------------------------------------------------------------------------------------------ INIT DNN INPUT PARAMETER
    self.input_size = 0
    self.num_classes = 0
    self._init_dnn_parameter()
    #-------------------------------------------------------------------------------------------------- OUTPUT PARAMETER
    self.output_filename = None
    self.output_file = None
    self.csv_writer = None
    #-------------------------------------------------------------------------------------------------- FEATURE SELECTOR
    self.feature_selector = None
    #--------------------------------------------------------------------------------- DEFINE DNN VARIABLES AND BUILD IT
    self.sess = None
    self.x = None
    self.y_ = None
    self.keep_prob_input = None
    self.keep_prob_hidden = None
    self.logits = None
    self.train_step = None
    self.correct_prediction_tr = None
    self.accuracy_tr = None

    # class weighting
    self.class_weights = None

    self._build_dnn()

    self.train_dataset = None
    self.test_dataset = None
    self.class_weighting_block_train = None
    self.class_weighting_block_test = None

    self.result = {}
    self.pool = multiprocessing.Pool(NUM_PROCESSES)  # process pool for preprocessing of a data block 

    self.connect()
    self.start()  # start the connection which use the overwritten process method to process the received data

    self._train()  # start training and testing loop

  def arg_parse(self):
    ''' create command line parser '''
    parser = defaultParser.create_default_parser()
    self.FLAGS, _ = parser.parse_known_args()
    defaultParser.printFlags(self.FLAGS)

  def _init_dnn_parameter(self):
    ''' evaluate input parameter for the DNN configuration '''
    num_features = 0
    for feature_str in self.FLAGS.features:
      feature = feature_str.split('-')
      feature_type_len = feature_selector.get(int(feature[0]))
      if not feature_type_len:
        raise Exception('invalid feature selection')
      feature_len = feature_type_len[int(feature[1])]
      if feature_len == -1:
        raise Exception('invalid feature type')
      num_features += feature_len

    self.input_size = num_features
    self.num_classes = len(self.FLAGS.boundaries_bps)

  def feed_dict(self, test=False, shuffle=False):
    ''' feed dict function '''
    if not test:
      xs, ys = self.train_dataset.next_batch(self.FLAGS.batch_size)
      k_h = self.FLAGS.dropout_hidden
      k_i = self.FLAGS.dropout_input
    else:
      xs, ys = self.test_dataset.next_batch(self.FLAGS.batch_size, shuffle=shuffle)
      k_h = 1.0
      k_i = 1.0

    feed_dict_ = {
        self.x: xs,  # data
        self.y_: ys,  # labels
        self.keep_prob_input: k_i,  # dropout probability input layer
        self.keep_prob_hidden: k_h,  # dropout probability hidden layer
        }

    if self.FLAGS.cw_method == 0:  # udpate feed dict with class weights
      if not test:
        feed_dict_[self.class_weights] = self.class_weighting_block_train
      else:
        feed_dict_[self.class_weights] = self.class_weighting_block_test
    return feed_dict_

  def _build_dnn(self):
    ''' build TensorFlow fully-connected DNN graph

    1. load/define variables for DNN
    2. build DNN
    '''
    # region -------------------------------------------------------------------------- 1. load/define variables for DNN
    self.output_filename = self.FLAGS.output_file

    if self.FLAGS.features:
      self.feature_selector = self.FLAGS.features

    if self.FLAGS.output_file:  # create CSV writer
      self.output_file = open(self.output_filename, 'w')
      self.csv_writer = csv.writer(self.output_file)

    # start an interactive session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    self.sess = tf.InteractiveSession(config=config)

    # placeholder for input variables
    self.x = tf.placeholder(tf.float32, shape=[None, self.input_size])
    self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes])
    self.keep_prob_input = tf.placeholder(tf.float32)
    self.keep_prob_hidden = tf.placeholder(tf.float32)

    self.class_weights = tf.placeholder(tf.float32, shape=[self.num_classes])  # only if class weighting is used
    # endregion

    # region --------------------------------------------------------------------------------------- dnn build functions
    def weight_variable(shape, stddev):
      ''' create weight variables for a whole layer '''
      initial = tf.truncated_normal(shape, stddev=stddev)
      return tf.Variable(initial)

    def bias_variable(shape):
      ''' create bias variables for a whole layer '''
      initial = tf.zeros(shape)
      return tf.Variable(initial)

    def fc_layer(x, channels_in, channels_out, stddev):
      ''' create a whole layer W * x + b with ReLU transfer function '''
      W = weight_variable([channels_in, channels_out], stddev)
      b = bias_variable([channels_out])
      act = tf.nn.relu(tf.matmul(x, W) + b)
      return act

    def logits_fn(x, channels_in, channels_out, stddev):
      ''' create a whole output layer W * x + b '''
      W = weight_variable([channels_in, channels_out], stddev)
      b = bias_variable([channels_out])
      act = tf.matmul(x, W) + b
      return act
    # endregion

    # region ---------------------------------------------------------------------------------------------- 2. build DNN
    x_drop_inn = tf.nn.dropout(self.x, self.keep_prob_input)

    # input layer
    h_fc_prev = fc_layer(x_drop_inn, self.input_size, self.FLAGS.layers[0], stddev=1.0 / math.sqrt(float(self.input_size)))
    h_fc_prev = tf.nn.dropout(h_fc_prev, self.keep_prob_hidden)

    for l, l_size in enumerate(self.FLAGS.layers[1:]):  # create hidden layers based on command line parameters
      h_fc_prev = fc_layer(h_fc_prev, self.FLAGS.layers[l], l_size, 1.0 / math.sqrt(float(self.FLAGS.layers[l])))
      h_fc_prev = tf.nn.dropout(h_fc_prev, self.keep_prob_hidden)

    # create output layer (a softmax linear classification layer_sizes for the outputs)
    self.logits = logits_fn(h_fc_prev, self.FLAGS.layers[-1], self.num_classes, stddev=1.0 / math.sqrt(float(self.FLAGS.layers[-1])))

    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.logits)
    if self.FLAGS.cw_method == 0:  # apply class weighting based on a weighting factor of a block
      weight_factor = tf.gather(self.class_weights, tf.cast(tf.argmax(self.y_, 1), tf.int32))
      avg_loss = tf.reduce_mean(cross_entropy_loss * weight_factor)
    else:
      avg_loss = tf.reduce_mean(cross_entropy_loss)  # no class weighting

    self.train_step = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(avg_loss)

    self.correct_prediction_tr = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
    self.accuracy_tr = tf.reduce_mean(tf.cast(self.correct_prediction_tr, tf.float32))
    # endregion

    tf.global_variables_initializer().run()  # initialize all global variables

  def _train(self):
    ''' training and testing loop (log testing results to file)

    * check after each batch if new data is available
    * train step
    * test step (based on log frequency)
    '''
    # region -------------------------------------------------------------------------------------- training and testing
    while True:  # load block, training and testing loop
      if len(self.result) == 0:
        time.sleep(1)
        continue

      # load preprocessed block
      self.train_dataset = self.result.get('train')
      self.test_dataset = self.result.get('test')
      self.class_weighting_block_train = self.result.get('class_weighting_train')
      self.class_weighting_block_test = self.result.get('class_weighting_test')
      self.result.clear()

      batch = 0

      batches_per_epoch_test = math.ceil(self.test_dataset.images.shape[0] / self.FLAGS.batch_size)  # how many batches per test epoch

      while True:
        if len(self.result) > 0:  # if next data block is available
          if self.csv_writer:
            self.output_file.flush()
          break  # stop training and testing -> load next block
        self.sess.run(self.train_step, feed_dict=self.feed_dict())  # training

        if batch % self.FLAGS.log_frequency == 0:  # testing after a defined number of batches
          acc = 0.
          for _ in range(batches_per_epoch_test):  # test for a full test epoch
            acc += self.sess.run(self.accuracy_tr, feed_dict=self.feed_dict(test=True))
          acc /= batches_per_epoch_test
          print('test accuracy at block({}) in step({}): {}'.format(self.block, batch, acc))
          if self.csv_writer: self.csv_writer.writerow([self.block, batch, acc])

        batch += 1
      self.block += 1    
    # endregion

    if self.csv_writer:
      output_file.close()

  def process(self, data):
    ''' start the preprocessing step for the next incoming block (called by base class)

    @param data: input data from stream server (np.array)
    @return: None (never stop the processing)
    '''
    Flow_Processor(data, self.result, self.FLAGS, self.pool).start()


if __name__ == '__main__':
  client = Stream_Client_FC_DNN()
  client.join()
