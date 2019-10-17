'''
file streaming client to replay stored datasets
'''
from log import init_logger
if __name__ == '__main__':  # only init logger for main process
  init_logger(__file__)
  
import client
import defaultParser
from datetime import datetime
from time import time
import os
import pickle
import gzip
from dateutil import parser as date_parser
from processor import Flow_Processor
import multiprocessing
from constants import NUM_PROCESSES
from log import log

units = {
  'B' : 1,
  'KB': 2 ** 10,
  'MB': 2 ** 20,
  'GB': 2 ** 30,
  }


class Stream_Client_File(client.Stream_Client):
  ''' stream client with file writer (pickels data into file)

  format: each block (dict('data':data, 'properties':dict(...))) is appended to file
  '''

  def __init__(self, output_file,
               iterations=None,
               stop_time=None,
               file_size=(1, 'GB'),
               ):
    ''' initialize the stream client for writing into files

    @param output_file: output filename (str)
    @param iterations: (optional) stop after a number of iterations (int)
    @param stop_time: (optional) timestamp when to stop (str) format: 'Apr 18 2019 10:37AM'
    @param file_size: (optional) stop after file size in unit (B, KB, MB, GB) exceeds (int, str) default: 1 GB
    '''
    client.Stream_Client.__init__(self)
    
    self.block = 0
    self.FLAGS = None
    self.arg_parse()
    
    self.start_timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')

    self.output_file = output_file
    self.iterations = iterations
    self.current_iteration = 0

    self.stop_time = date_parser.parse(stop_time) if stop_time else None
    self.file_size_limit = file_size[0] * units.get(file_size[1]) if file_size else None

    self.file = None
    self.file_op = self._filename(output_file)
    
    self.result = {}
    self.pool = multiprocessing.Pool(NUM_PROCESSES)  # process pool for preprocessing of a data block

  def arg_parse(self):
    ''' create command line parser '''
    parser = defaultParser.create_default_parser()
    self.FLAGS, _ = parser.parse_known_args()
    # defaultParser.printFlags(self.FLAGS)

  def _filename(self, filename):
    ''' return filename + filemode (create or append) (optional: append file extension)

    @param filename: filename (str)
    @return: extended filename (str), file write mode (str)
    '''
    if not filename.endswith('.pkl.gz'):
      filename += '.pkl.gz'

    self.file = filename
    if os.path.isfile(filename):
      return self.file, 'ab'

    return self.file, 'wb'

  def _pickle_data(self, data, filename):
    ''' write data to a pickle file '''
    file_op = self._filename(filename)

    with gzip.GzipFile(*file_op) as file:
      pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

  def process(self, data):
    ''' preprocess and store data until:
      * a defined timestamp
      * a defined file size
      * a defined number of blocks

    @param data: block of data
    @return: training should be stopped (True) or not (False)
    '''
    if (# exit on defined time
      self.stop_time  # stop time is set
      and datetime.now() >= self.stop_time  # stop time exceeds
      ):
      print('exit because stop_time ({}) exceeds'.format(self.stop_time))
      return True

    if (# exit after x iterations
      self.iterations  # iteration limit is set
      and self.current_iteration >= self.iterations  # iteration limit reached
      ):
      print('exit because iterations limit ({}) reached'.format(self.iterations))
      return True

    if (# exit if file size limit is reached
      self.file_size_limit  # file limit set
      and self.file  # file name set
      and os.path.isfile(self.file)  # file exists
      and os.path.getsize(self.file) >= self.file_size_limit
      ):
      print('exit because file size ({}/{}) exceeds'.format(os.path.getsize(self.file), self.file_size_limit))
      return True

    processor_thread = Flow_Processor(data, self.result, self.FLAGS, self.pool)
    processor_thread.run()

    save = {
      'dataset_train': self.result.get('train'),
      'dataset_test': self.result.get('test'),
      'properties': {'timestamp': datetime.now()},
    }
    log.debug('pickling block {}'.format(self.block))
    self._pickle_data(save, self.output_file)
    self.current_iteration += 1
    self.block += 1
    
    self.result.clear()

    return False  # start next iteration


if __name__ == '__main__':
  scf = Stream_Client_File('data',
                           file_size=None,
                           # file_size=(5, 'KB'),
                           # iterations=50,
                           # stop_time='Apr 18 2019 10:37AM'
                           )
  scf.connect()
  scf.start()
  scf.join()
