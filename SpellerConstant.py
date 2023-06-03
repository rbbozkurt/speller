FIREBASE_REFERENCE = 'https://speller-2ead4-default-rtdb.europe-west1.firebasedatabase.app/'
DEVICES_NODE_REFERENCE = '/connected_devices'
SENSOR_AXES = ['game_rotation_vector_U', 'game_rotation_vector_X', 'game_rotation_vector_Y', 'game_rotation_vector_Z',
               'gyroscope_X', 'gyroscope_Y', 'gyroscope_Z', 'linear_acceleration_X', 'linear_acceleration_Y',
               'linear_acceleration_Z']
PLOT_SAVE_PATH = 'figures/'
#SUMMARY_STATISTICS_FEATURES = ['MIN', 'MAX', 'PEAK_INF0', 'DIP_INF0', 'PEAK_INF1', 'DIP_INF1', 'HJ_MOB', 'HJ_COM',
         #                      'MEAN', 'VAR', 'STD', 'RMS','POWER', 'PEAK', 'P2P', 'CREST_FACTOR', 'SKEW', 'KURTOSIS',
          #                     'MAX_f', 'SUM_f', 'MEAN_f', 'VAR_f', 'PEAK_f', 'SKEW_f', 'KURTOSIS_f']

SUMMARY_STATISTICS_FEATURES = ['MIN', 'MAX', 'PEAK_INF0', 'DIP_INF0', 'PEAK_INF1', 'DIP_INF1', 'HJ_MOB', 'HJ_COM',
                               'MEAN', 'VAR', 'STD']
PEAK_DETECTOR_LAG = 2
