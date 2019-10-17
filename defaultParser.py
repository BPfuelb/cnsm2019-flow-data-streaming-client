'''
default command line parameter parser
'''
import argparse


def printFlags(flags):
    ''' print all command line parameters '''
    side = '-' * int(((80 - len('Flags')) / 2))
    print(side + ' ' + 'Flags' + ' ' + side)
    for k, v in sorted(vars(flags).items()):
      print('\t * {}: {}'.format(k, v))
    print('-' * 80)


def create_default_parser():
    ''' create a parser with default parameters for most experiments '''
    parser = argparse.ArgumentParser()

    #-------------------------------------------------------- DROPTOUT PARAMETER
    parser.add_argument('--dropout_hidden', type=float, default=1)
    parser.add_argument('--dropout_input', type=float, default=1)

    #----------------------------------------------------------- LAYER PARAMETER
    parser.add_argument('--layers', type=int, nargs='+', default=[1000, 1000, 1000])

    #----------------------------------- BATCH_SIZE, LEARNING_RATE, EPOCHS, ETC.
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--log_frequency', type=int, default=50)

    default_features = [
      '0-1',  # src_addr
      '1-1',  # dst_addr
      # '2-' # packets
      # '3-' # bytes
      # '4-' # first switched
      # '5-' # last switched
      '6-1',  # src_port
      '7-1',  # dst_port
      '8-1',  # tcp_flags
      '9-1',  # protocol
      # '10-' # export host
      # '11-' # flow seq number
      # '12-' # duration
      # '13-' # bitrate
      '14-0',  # src_country_code
      '15-0',  # src_longitude
      '16-0',  # src_latitude
      '17-1',  # src_asn
      '18-1',  # src_network
      '19-0',  # src_prefix_len
      '20-1',  # src_vlan
      '21-1',  # src_locality
      '22-0',  # dst_country_code
      '23-0',  # dst_longitude
      '24-0',  # dst_latitude
      '25-1',  # dst_asn
      '26-1',  # dst_network
      '27-0',  # dst_prefix_len
      '28-1',  # dst_vlan
      '29-1',  # dst_locality
      # '30-' # year
      '31-0',  # month
      '32-0',  # day
      '33-0',  # hour
      '34-0',  # minute
      '35-0',  # second
  ]
    #------------------------------------------------------------------ FEATURES
    parser.add_argument('--features', type=str, nargs='+', default=default_features,
                        help='select features for training and testing')
    parser.add_argument('--cw_method', type=int, default=0,
                        help='0 = standard class weighting; 1 = under-sampling')
    parser.add_argument('--feature_filter', type=str, nargs='*', default='',
                        help=('set filter functions: "feature_key;lambda x: <bool> "'
                        '  e.g.,["(6,);lambda x: x == 53.", "(7,);lambda x: x == 53."]'
                        '  or ["(6,7);lambda x,y: x == 53. or y == 53."]'))

    parser.add_argument('--boundaries_bps', type=float, nargs='+', default=[0., 50., 8000.],
                        help='set boundaries for bps')
    parser.add_argument('--boundaries_duration', type=float, nargs='+', default=[0., 100., 200.],
                        help='set boundaries for duration')

    #----------------------------------------------------- CSV OUTPUTS FOR PLOTS
    parser.add_argument('--output_file', type=str, default='out.csv')

    return parser
