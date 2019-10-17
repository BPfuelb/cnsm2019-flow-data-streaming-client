'''
processing unit to prepare flow data for model training and testing (side: flow data streaming client)
'''
import threading
from scipy.interpolate import interp1d
from utils import *
import math
from constants import *
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet as TF_DataSet
from functools import partial


class VALUE_RANGE():
  ''' normalization value range '''
  min = 0.0
  max = +1.0


class Feature():
  ''' enumeration class defines positions of features '''
  # DATA
  SRC_ADDR = 0
  DST_ADDR = 1
  PACKETS = 2  # (possible) label
  BYTES = 3  # (possible) label
  FIRST_SWITCHED_TS = 4
  LAST_SWITCHED_TS = 5
  SRC_PORT = 6
  DST_PORT = 7
  TCP_FLAGS = 8
  PROTOCOL = 9
  EXPORT_HOST = 10
  FLOW_SEQ_NUM = 11
  DURATION = 12  # (possible) label
  BIT_RATE = 13  # (possible) label
  # LOOKUP SRC
  SRC_COUNTRY_CODE = 14
  SRC_LONGITUDE = 15
  SRC_LATITUDE = 16
  SRC_ASN = 17
  SRC_NETWORK = 18
  SRC_PREFIX_LEN = 19
  SRC_VLAN = 20
  SRC_LOCALITY = 21
  # LOOKUP DST
  DST_COUNTRY_CODE = 22
  DST_LONGITUDE = 23
  DST_LATITUDE = 24
  DST_ASN = 25
  DST_NETWORK = 26
  DST_PREFIX_LEN = 27
  DST_VLAN = 28
  DST_LOCALITY = 29
  # FIRST SWITCHED TIMESTAMP
  FS_YEAR = 30
  FS_MONTH = 31
  FS_DAY = 32
  FS_HOUR = 33
  FS_MINUTE = 34
  FS_SECOND = 35


FIVE_TUPLE = [  # definition of 5-tuple features
  Feature.PROTOCOL,
  Feature.SRC_ADDR,
  Feature.DST_ADDR,
  Feature.SRC_PORT,
  Feature.DST_PORT,
  ]

country_abbreviations = None  # placeholder for country abbreviation dictionary


class Datatype():
  ''' defines datatype conversion methods '''
  FLOAT = 0
  BITS = 1
  ONE_HOT = 2


class Timestamp():
  ''' value positions in timestamp '''
  YEAR = 0
  MONTH = 1
  DAY = 2
  HOUR = 3
  MINUTE = 4
  SECOND = 5


def norm_value(data_type, data, **kwargs):
  ''' normalization function (additionally create an inverse function of itself)
    e.g. performed normalization of a value, e.g., port=3500 to port=0.053

    @param data_type: defines data type conversion method (Datatype) (e.g. BITS, FLOAT, ONE-HOT)
    @param data: values (column) that get converted (np.array)
    @param kwargs:
    'input_type' defines the raw data input type (e.g., np.uint32)
    'min_' and 'max_' defines the range of an value (e.g., [0,59] for miniutes)
    'output_type' defines output type of values (e.g., np.uint8, extra IPv4Address)
  '''
  min_range = VALUE_RANGE.min
  max_range = VALUE_RANGE.max

  # region ----------------------------------------------------------------------------------------------- Datatype.BITS
  if data_type == Datatype.BITS:
    input_type = kwargs.get('input_type')
    min_ = kwargs.get('min_', 0.)  # if not set, minimum is 0
    max_ = kwargs.get('max_', np.iinfo(input_type).max)  # if not set, maximum is determined based on input type

    bits = np.iinfo(input_type).bits  # get number of used bits for input type
    data = data.astype(input_type)  # convert input data to input data type
    data = data.view(np.uint8)  # byte view of the data (needed for unpackbits)
    data = np.unpackbits(data, axis=0)  # convert byte to single bits
    data = data.reshape(-1, 8)  # reshape back to numpy bytes array e.g., [0 1 1 0 1 0 1 0], ...

    # reverse sequence of bytes for np.uint16 and np.uint32
    if input_type == np.uint16:
      data[1::2], data[0::2] = data[::2].copy(), data[1::2].copy()
    if input_type == np.uint32:
      data[3::4], data[2::4], data[1::4], data[::4] = data[::4].copy(), data[1::4].copy(), data[2::4].copy(), data[3::4].copy()

    data = data.reshape(-1, bits)  # reshape bytes to number of input type bits
    data = data.astype(np.float32)  # convert as bits as floats

    num_values = len(range(int(min_), int(max_) + 1))  # calculate number of bits with min_ and max_ values
    slice_ = math.ceil(math.log(num_values, 2))  # calculate number of bits with num_values
    if slice_ < bits:  # if a slice_ is smaller num bits (e.g., for VLAN only 12 bits needed)
      data = data[:, -slice_:]

    data[data == 0.0] = min_
  # endregion
  # region ---------------------------------------------------------------------------------------------- Datatype.FLOAT
  if data_type == Datatype.FLOAT:
    min_ = kwargs.get('min_', 0.0)
    input_type = kwargs.get('input_type')
    max_ = kwargs['max_'] if 'max_' in kwargs else np.iinfo(input_type).max
    output_type = kwargs.get('output_type')

    # region ------------------------------------------------------------------------------------------------ convert IP
    # e.g., A.B.C.D -> float(A), float(B), float(C), float(D)
    if output_type == np.uint32:  # only used for IP addresses
      data = data.astype(output_type)  # convert to output type
      split_oct_type = np.dtype((np.int32, {  # define split data type
        'oct0':(np.uint8, 3),
        'oct1':(np.uint8, 2),
        'oct2':(np.uint8, 1),
        'oct3':(np.uint8, 0),
        }))
      data = data.view(dtype=split_oct_type)  # apply split data type

      # create and apply octet convert function on each octet
      data_fn = np.vectorize(interp1d([0, np.iinfo(np.uint8).max], [min_range, max_range]))
      oct0 = data_fn(data['oct0'])
      oct1 = data_fn(data['oct1'])
      oct2 = data_fn(data['oct2'])
      oct3 = data_fn(data['oct3'])

      data = np.array([oct0, oct1, oct2, oct3]).T  # recombine converted octets
    # endregion
    data_fn = np.vectorize(interp1d([min_, max_], [min_range, max_range]))
    data = data_fn(data)
  # endregion
  # region -------------------------------------------------------------------------------------------- Datatype.ONE_HOT
  if data_type == Datatype.ONE_HOT:
    min_ = kwargs.get('min_')
    max_ = kwargs.get('max_')

    for_all = np.arange(data.shape[0])
    data = data.astype(np.int32)
    value_as_index = data - min_  # shift all values based on minimum (e.g., month -> 1 => 0; 12 => 11)

    data = np.full((data.shape[0], max_ - min_ + 1), min_range, dtype=np.float32)  # define new one hot vector
    data[for_all, value_as_index] = max_range  # set value in one hot (month = 0 = [ 1 0 ... ])
    data = np.fliplr(data)  # flip value (month = 0 = [ ... 0 1 ])
  # endregion
  return data


class Flow_Processor(threading.Thread):

  def __init__(self, data, result, FLAGS, pool):
    ''' processing class that uses a process pool to distribute a data block as chunks and collect the results of the executed preprocessing steps
        (side: flow data streaming client)
    @param data: input flow data block to process (np.array)
    @param result: reference to result dict (dict)
    @param FLAGS: parsed command line parameters (argparse.Namespace)
    @param pool: process pool (multiprocessing.Pool)
    '''
    threading.Thread.__init__(self)

    self.FLAGS = FLAGS
    self.data = data
    self.result = result
    self.features = { int(feature_dtype.split('-')[0]): int(feature_dtype.split('-')[1]) for feature_dtype in FLAGS.features }
    self.feature_filter = {}

    self.main_result_queue = multiprocessing.Manager().Queue()  # synchronized queue for results
    self.pool = pool

    for filter_expression in FLAGS.feature_filter:  # '--feature_filter "1 ; lambda x: x < 10" "2 ; lambda y : y > 5"'
      if filter_expression == '': break
      exp = filter_expression.split(';')[1]
      feature_key = eval(filter_expression.split(';')[0])
      self.feature_filter.update({feature_key: exp})
    # print_feature_filter_list(self.feature_filter)

  def run(self):
    ''' start the preprocessing of a flow data block

      * split the data and distribute it to the processes in the process pool
      * wait until all processes are finished and combine the results
      * (optional) calculate class weighting factors or apply undersampling
      * provide the data to the machine learning model (DNN)
    '''
    start_process_block = time.time()
    data = self.data
    
    # split data -> data chunks
    data = np.array_split(data, NUM_PROCESSES)

    # distribute the data chunks to the processes
    args = [ (pid_, data_slice, self.FLAGS, self.features, self.feature_filter, self.main_result_queue) for pid_, data_slice in enumerate(data) ]
    self.pool.map(bar, args)

    # collect the results
    results = {}
    if self.main_result_queue.qsize() != NUM_PROCESSES: raise Exception('result queue != NUM_PROCESSES')
    for _ in range(self.main_result_queue.qsize()):
      element = self.main_result_queue.get()
      results[element[0]] = (element[1], element[2])
    data = [ results[id_][0] for id_ in range(NUM_PROCESSES) ]
    labels = [ results[id_][1] for id_ in range(NUM_PROCESSES) ]

    # combine the results
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    data_train, labels_train, data_test, labels_test = self._split(data, labels, 10)
    log.debug('class_distribution_labels_train: {}'.format(np.bincount(np.argmax(labels_train, axis=1))))
    log.debug('class_distribution_labels_test: {}'.format(np.bincount(np.argmax(labels_test, axis=1))))

    train_class_weighting = None
    test_class_weighting = None
    if self.FLAGS.cw_method == 0:  # standard class weighting
      class_counter = np.bincount(np.argmax(labels_train, axis=1))
      inverse = 1. / class_counter
      max_ = max(class_counter)
      train_class_weighting = inverse * max_
      train_class_weighting /= sum(train_class_weighting)

      class_counter = np.bincount(np.argmax(labels_test, axis=1))
      inverse = 1. / class_counter
      max_ = max(class_counter)
      test_class_weighting = inverse * max_
      test_class_weighting /= sum(test_class_weighting)

    if self.FLAGS.cw_method == 1:  # undersampling
      # apply undersampling on training data
      mask = np.zeros((labels_train.shape[0],), dtype=bool)
      min_count = min(np.bincount(np.argmax(labels_train, axis=1)))

      for current_class in range(3):
        class_indices = np.argmax(labels_train, axis=1).astype(np.int)
        current_class_indices = np.where(class_indices == current_class)[0][0:min_count]
        mask[current_class_indices] = True

      data_train = data_train[mask]
      labels_train = labels_train[mask]

      # apply undersampling on test data
      mask = np.zeros((labels_test.shape[0],), dtype=bool)
      min_count = min(np.bincount(np.argmax(labels_test, axis=1)))

      for current_class in range(3):  # TODO: variable number of classes
        class_indices = np.argmax(labels_test, axis=1).astype(np.int)
        current_class_indices = np.where(class_indices == current_class)[0][0:min_count]
        mask[current_class_indices] = True

      data_test = data_test[mask]
      labels_test = labels_test[mask]

    data_train, labels_train = self._shuffle(data_train, labels_train)
    data_test, labels_test = self._shuffle(data_test, labels_test)

    train = self._create_tf_dataset(data_train, labels_train)
    test = self._create_tf_dataset(data_test, labels_test)

    # provide results data to the training and testing process
    self.result.update(
      {'train': train,
       'test':  test,
       'class_weighting_train': train_class_weighting if self.FLAGS.cw_method == 0 else None,
       'class_weighting_test': test_class_weighting if self.FLAGS.cw_method == 0 else None,
      })

    log.debug('process_time_block: {}s'.format(time.time() - start_process_block))

  @measure_time_memory
  def _split(self, data, labels, percent_test):
    ''' split the input data and labels into training data and labels and test data and labels by a given percentage

    @param data: input data (np.array)
    @param labels: input labels (np.array)
    @param percent_test: percent value (int) of the amount of test data/lables (default = 10%)
    @return: data_train (np.array), labels_train (np.array), data_test (np.array), labels_test (np.array)
    '''
    num_elements_train = int(data.shape[0] * ((100 - percent_test) / 100))

    data_test = data[num_elements_train:]
    label_test = labels[num_elements_train:]

    data_train = data[:num_elements_train]
    labels_train = labels[:num_elements_train]

    return data_train, labels_train, data_test, label_test

  @measure_time_memory
  def _shuffle(self, data, labels):
    ''' shuffle data and labels

    @param data: input data (np.array)
    @param labels: input labels (np.array)

    @return: shuffled data (np.array), shuffled labels (np.array)
    '''
    permutation = np.random.RandomState(seed=42).permutation(data.shape[0])
    data = data[permutation]
    labels = labels[permutation]
    return data, labels

  @measure_time_memory
  def _create_tf_dataset(self, data, labels):
    return TF_DataSet(255. * data, labels, reshape=False)


def bar(args):
  ''' filter features, create labels and normalize data chunk-wise (each chunk is handled by one process of the process pool)
  
  @param args: list of arguments
    * [0]: id of the data chunk (int)
    * [1]: data (np.array)
    * [2]: command line parameters (argparse.Namespace)
    * [3]: selected features (list)
    * [4]: feature filters (dict)
    * [5]: reference of the result queue (synchronized)
   '''
  id_ = args[0]
  data = args[1]
  flags = args[2]
  features = args[3]
  feature_filter = { feature_key:eval(exp) for feature_key, exp in args[4].items() }
  result_queue = args[5]

  log.debug('number_of_elements_before_filtering: {}'.format(data.shape[0]))
  data = filter_features(data, feature_filter)
  log.debug('number_of_elements_after_filtering: {}'.format(data.shape[0]))

  labels = create_labels(data, flags)

  elements_per_class = np.bincount(np.argmax(labels, axis=1))
  log.debug('number_of_elements_per_class: {}'.format(elements_per_class))

  if data.size != 0:
    data = normalize(data, features)

  result_queue.put((id_, data, labels))


@measure_time_memory
def filter_features(data, feature_filter):
  ''' feature filter based on defined lambda expression(s)

  @param data: input data (np.array)
  @param feature_filter: feature filter dict (dict)
  @return: filtered data (np.array)
  '''
  for feature_key, lambda_exp in feature_filter.items():
    mask = np.squeeze(np.vectorize(lambda_exp)(*data[:, feature_key].T))
    mask = np.invert(mask)
    data = data[mask]
  return data


@measure_time_memory
def create_labels(data, flags):
  ''' convert class labels to one hot

  @param data: input data, all potential labels (np.array)
  @return classes in one hot format, number of classes (np.array)
  '''
  class_dims = []
  class_collection = []

  def create_label(data, bins):
    ''' classify the values into the individual classes

    @param data: input data, labels (np.array)
    @param bins: list of boundaries (list)
    '''
    class_dims.append(len(bins) - 1)
    data_ = data.astype(np.float32)
    classes_ = np.digitize(data_, bins) - 1
    class_collection.append(classes_)

  create_label(data[:, Feature.BIT_RATE], flags.boundaries_bps + [float('inf')])  # only use bit rate (bps) for class labeling
  ''' example: create 3 x 3 classes for labeling '''
  # create_label(data[:, Feature.DURATION], flags.boundaries_duration + [float('inf')])

  classes = np.zeros((data.shape[0], *class_dims))  # create class matrix with selected label dimensions

  index = np.arange(len(data))
  classes[index, (*class_collection)] = 1  # mark selected class
  _num_classes = np.prod(class_dims)
  classes = np.reshape(classes, (len(data), _num_classes))  # reshape to one hot

  return classes.astype(np.float32)


@measure_time_memory
def normalize(data, features):
  ''' normalize data

    @param data: all data values (np.array)
    @return: normalized data [first_switched, protocol, ip, port, pref_len, asn, geo_coordinates, country_code,
                              vlan, locality, tcp_flags,], class labels [duration, bps, cor_duration, cor_bps]
                              (np.array, np.array)
  '''

  def interpolate(min_range, max_range, data):
    ''' helper function for the interpolation of float values with predefined range values

    @param min_range: lower limit for interpolation (float)
    @param max_range: upper limit for interpolation (float)
    @param data: input data (np.array)
    @return: interpolated value (float)
    '''
    return interp1d([min_range, max_range], [VALUE_RANGE.min, VALUE_RANGE.max])(data)

  def norm_fn(data, data_type, column, float_=True, bits=True, one_hot=True, output_type=np.uint8, input_type=np.uint8, **kwargs):
    ''' wrapper for norm_value function

    @param data: full input data (np.array)
    @param data_type: conversion type (Datatype)
    @param column: data column (Feature)
    @param float_: is convertible to float value (bool)
    @param bits: is convertible to bit pattern (bool)
    @param one_hot: is convertible to one hot (bool)
    @param output_type: data type for return values (np.dtype)
    @param input_type: data type of input values (np.dtype)
    @param **kwargs: multiple additional parameter (see norm_value function)
    @return: normalized data (np.array)
    '''
    # define not possible conversion types
    if not float_  and data_type == Datatype.FLOAT:
      raise NotImplementedError()
    if not bits    and data_type == Datatype.BITS:
      raise NotImplementedError()
    if not one_hot and data_type == Datatype.ONE_HOT:
      raise NotImplementedError()

    output = norm_value(data_type, data[:, column], output_type=output_type, input_type=input_type, **kwargs)
    output = np.array(output)

    # optional transform vector
    if data_type == Datatype.FLOAT:
      return output[np.newaxis].T
    if data_type == Datatype.BITS:
      return output
    if data_type == Datatype.ONE_HOT:
      return output

  # partial bind normalization function
  norm_functions = [
    partial(norm_fn, column=Feature.SRC_ADDR, one_hot=False, output_type=np.uint32, input_type=np.uint32),
    partial(norm_fn, column=Feature.DST_ADDR, one_hot=False, output_type=np.uint32, input_type=np.uint32),
    None,  # packets
    None,  # bytes
    None,  # first_switched
    None,  # last_switched
    partial(norm_fn, column=Feature.SRC_PORT, one_hot=False, output_type=np.uint16, input_type=np.uint16),
    partial(norm_fn, column=Feature.DST_PORT, one_hot=False, output_type=np.uint16, input_type=np.uint16),
    partial(norm_fn, column=Feature.TCP_FLAGS, one_hot=False),
    partial(norm_fn, column=Feature.PROTOCOL),
    None,  # export host
    None,  # FLOW_SEQ_NUM
    None,  # duration
    None,  # bit rate
    partial(norm_fn, column=Feature.SRC_COUNTRY_CODE, min_=0, max_=250 - 1),
    partial(norm_fn, column=Feature.SRC_LONGITUDE, one_hot=False, bits=False, min_=-90., max_=90., output_type=np.float32, input_type=np.float32),
    partial(norm_fn, column=Feature.SRC_LATITUDE, one_hot=False, bits=False, min_=-180., max_=180., output_type=np.float32, input_type=np.float32),
    partial(norm_fn, column=Feature.SRC_ASN, one_hot=False, output_type=np.uint16, input_type=np.uint16),
    partial(norm_fn, column=Feature.SRC_NETWORK, one_hot=False, output_type=np.uint32, input_type=np.uint32),
    partial(norm_fn, column=Feature.SRC_PREFIX_LEN, one_hot=False, min_=0, max_=32),
    partial(norm_fn, column=Feature.SRC_VLAN, one_hot=False, min_=0, max_=4095, output_type=np.uint16, input_type=np.uint16),
    partial(norm_fn, column=Feature.SRC_LOCALITY, float=False, one_hot=False, min_=0, max_=1),
    partial(norm_fn, column=Feature.DST_COUNTRY_CODE, min_=0, max_=250 - 1),
    partial(norm_fn, column=Feature.DST_LONGITUDE, one_hot=False, bits=False, min_=-90., max_=90., output_type=np.float32, input_type=np.float32),
    partial(norm_fn, column=Feature.DST_LATITUDE, one_hot=False, bits=False, min_=-180., max_=180., output_type=np.float32, input_type=np.float32),
    partial(norm_fn, column=Feature.DST_ASN, one_hot=False, output_type=np.uint16, input_type=np.uint16),
    partial(norm_fn, column=Feature.DST_NETWORK, one_hot=False, output_type=np.uint32, input_type=np.uint32),
    partial(norm_fn, column=Feature.DST_PREFIX_LEN, one_hot=False, min_=0, max_=32),
    partial(norm_fn, column=Feature.DST_VLAN, one_hot=False, min_=0, max_=4095, output_type=np.uint16, input_type=np.uint16),
    partial(norm_fn, column=Feature.DST_LOCALITY, float=False, one_hot=False, min_=0, max_=1),
    None,  # fs_year
    partial(norm_fn, column=Feature.FS_MONTH, min_=1., max_=12.),
    partial(norm_fn, column=Feature.FS_DAY, min_=1., max_=31.),
    partial(norm_fn, column=Feature.FS_HOUR, min_=0., max_=23.),
    partial(norm_fn, column=Feature.FS_MINUTE, min_=0., max_=59.),
    partial(norm_fn, column=Feature.FS_SECOND, min_=0., max_=59.),
    ]

  # apply partially normalization functions for all selected features
  normed_features = []
  for feature_key, feature_datatype in features.items():
    norm_function = norm_functions[feature_key]
    if not norm_function:
      raise Exception('invalid feature selection')
    normed_data = norm_function(data, feature_datatype)
    normed_features.append(normed_data)

  data = np.concatenate(tuple(normed_features), axis=1).astype(np.float32)  # recombine normalized features

  return data
