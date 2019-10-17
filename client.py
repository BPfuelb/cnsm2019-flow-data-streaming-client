'''
base class for streaming clients

represent a streaming client that implements the Normalization step of the Data Preparation stage and the Machine Learning stage
'''

from constants import *
import pickle
import zlib
from abc import ABC, abstractmethod
from socket import AF_INET, SOCK_STREAM, socket
import threading
import itertools
import struct
from log import log


class Stream_Client(ABC, threading.Thread):

  def __init__(self):
    threading.Thread.__init__(self)
    self.socket = None
    self.running = True
    self.num_socket_timeouts = 0

  @abstractmethod
  def process(self, data):
    ''' abstract process method is called by the thread itself (in run)

    @param data: input data from stream server
    @return: if true, thread is stopped (bool), else: stop processing and disconnect from server (only used by file client)
    '''
    pass

  def connect(self):
    ''' use constants to create a connect to the streaming server '''
    log.info('connect to {}:{}'.format(SERVER_ADDRESS, SERVER_PORT))
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.settimeout(SOCKET_TIMEOUT)
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))
    self.socket = client_socket

  def stop(self):
    ''' stop the processing loop '''
    self.running = False

  def _disconnect(self):
    ''' close the connection to the streaming server '''
    log.info('exit')
    self.socket.close()

  def run(self):
    ''' loop for receiving data and start the processing step '''
    while self.running:
      log.debug('wait for data...')
      all_data = []
      data = self.socket.recv(1500)
      num_bytes = struct.unpack('!I', data[:4])[0]
      data = data[4:]
      all_data.append(data)
      num_bytes -= len(data)
      while num_bytes:  # load data until a full data block was received
        data = self.socket.recv(1500)
        all_data.append(data)
        num_bytes -= len(data)

      log.debug('unpack data...')
      all_data_flat = list(itertools.chain(*all_data))  # concatenate all received lists
      data = zlib.decompress(bytes(all_data_flat))  # unzip data
      data = pickle.loads(data)  # unpickle data
      log.debug('data.shape: {}'.format(data.shape))
      if self.process(data):  # start data processing
        break

    self._disconnect()
