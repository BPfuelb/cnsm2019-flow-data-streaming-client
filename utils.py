'''
load/install modules
helper functions (print functions)
'''


def load_module(module, package=None):
  ''' auto module loader, if module not present, install into user space via pip

  @param module: module name
  @param package: (optional) package name to install if different module name
  '''
  try:
    import pip
    if hasattr(pip, 'main'):
      from pip import main as pip_install  # linux
    else:
      from pip._internal import main as pip_install  # windows
  except:
    raise Exception('install pip')
  try:
    import importlib
    importlib.import_module(module)
  except ImportError as ex:
    print(ex)
    if package is None:
      pip_install(['install', '--user', module])
    else:
      pip_install(['install', '--user', package])
  finally:
    globals()[module] = importlib.import_module(module)


# required modules
modules = [
  'numpy'      ,
  'sys'        ,
  'math'       ,
  'itertools'  ,
  'abc'        ,
  'argparse'   ,
  'collections',
  'functools'  ,
  'importlib'  ,
  'os'         ,
  'scipy'      ,
  'shutil'     ,
  'time'       ,
  'gzip'       ,
  'pickle'     ,
  'zipfile'    ,
  'tarfile'    ,
  'pathlib'    ,
  'urllib'     ,
  'datetime'   ,
  'ipaddress'  ,
  'builtins'   ,
  'enum'       ,
  'csv'        ,
  ]

# auto install routine for modules
for module in modules:
  load_module(module)
load_module('safe_cast', 'safe-cast')
# load_module('tensorflow', 'tensorflow-gpu==1.14') # gpu variant
load_module('tensorflow', 'tensorflow-gpu==1.14')  # cpu variant TODO: protobuf>=3.6.1 required

import numpy as np
import sys
import time
import psutil
import os
from log import log

tsne_output = None
block = 0


def measure_time_memory(method):
  ''' decorator for time and memory measurement of functions
  
  @param method: function for which the measurements are performed (func)
  @return result(s) of the execution of method
  '''

  def measure(*args, **kw):
    log_msg = method.__name__
    start = time.time()
    result = method(*args, **kw)
    if hasattr(psutil.Process(), 'memory_info'):
      mem = psutil.Process(os.getpid()).memory_info()[0] // (2 ** 20)
      log_msg += ': {:2.2f}s mem: {}MB'.format(time.time() - start, mem)
    elif hasattr(psutil.Process(), 'memory_full_info'):
      mem = psutil.Process(os.getpid()).memory_full_info()[0] // (2 ** 20)
      log_msg += ': {:2.2f}s mem: {}MB'.format(time.time() - start, mem)
    else:
      log_msg += ': {:2.2f}s '.format(time.time() - start)
    if sys.platform != 'win32': log.debug(log_msg)
    else:                       print(log_msg)
    return result

  return measure


class Labels():
  ''' define positions (column) of labels in data '''
  BYTES = 0
  DURATION = 1
  BPS = 2
  REV_BYTES = 3
  REV_DURATION = 4
  REV_BPS = 5


np.set_printoptions(threshold=sys.maxsize)  # print whole numpy arrays


def print_feature_filter_list(feature_filter):
  ''' print feature filter list '''
  filter_list = [ (k, v) for k, v in feature_filter.items() if v != None ]
  if len(filter_list) == 0:
    return
  print('*' * 20, 'Feature Filter List', '*' * 20)
  for k, v in filter_list:
    print('key: {}  lambda expression: {}'.format(k, v))
  print('*' * 52)


def print_feature_filter_function_list(fn_list):
  ''' print feature filter function (partial functions) '''
  print('*' * 20, 'Feature Filter Function List', '*' * 20)
  for t in fn_list.items():
    print(t)
  print('*' * 52)
