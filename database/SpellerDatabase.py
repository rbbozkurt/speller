from firebase import firebase
import pandas as pd
import json
import SpellerConstant
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from pathlib import Path


def read_letters_from_npz(X_path, y_path):
    npz_X = np.load(X_path)
    npz_y = np.load(y_path)
    return npz_X['arr_0'], npz_y['arr_0']


def read_letters_from_npy(X_path, y_path):
    root_filename = Path("/Users/resitberkaybozkurt/PycharmProjects/speller_data")
    file_X = root_filename / X_path
    file_y = root_filename / y_path
    X = np.load(file_X.as_posix())
    y = np.load(file_y.as_posix())
    print(X.shape)
    print(y.shape)
    return X, y


@lru_cache()
def read_letters_from_database(firebase_ref, target_letters: tuple) -> pd.DataFrame():
    data_frame = pd.DataFrame()
    fb_app = firebase.FirebaseApplication(firebase_ref, None)
    devices_db = fb_app.get(SpellerConstant.DEVICES_NODE_REFERENCE, None)
    connected_devices_id = []
    for device in tqdm(devices_db.values(), desc="Getting connected devices"):
        device_dic = json.loads(device)
        connected_devices_id.append(device_dic['id'])
    for device_id in tqdm(connected_devices_id, desc="Reading database"):
        path = '/{}'.format(device_id)
        # print(path)
        device_letters = fb_app.get(path, None)
        if (device_letters != None):
            # print("Result is not None:\nPrinting letters:\n{}".format(result.keys()))
            labels = {}
            # for num, letter in enumerate(result.keys(), start=1):
            for num, letter in enumerate(target_letters, start=1):

                labels[letter] = float(num)
                letter_path = '{}/{}'.format(path, letter)
                record_date = fb_app.get(letter_path, None)
                if (record_date != None):
                    for date in record_date.keys():
                        new_row = {'date': date}
                        record_path = '{}/{}'.format(letter_path, date)
                        # print(record_path)
                        data = fb_app.get(record_path, None)
                        data_len = 0
                        for axis_ind, axis in enumerate(data.keys(), start=0):
                            data_seq = json.loads(data[axis])
                            data_len = len(data_seq)
                            np_array = np.array([])
                            length = 0
                            for ind, data_point in enumerate(data_seq, start=0):
                                np_array = np.append(np_array, float(data_point['value']))
                                # print('Axis: {}, Index: {}, Time: {}, Value: {}'.format(axis, ind, data_point['time'], data_point['value']
                                length = length + 1
                            new_row[axis] = np_array
                            if (axis_ind == 0):
                                new_row['data'] = new_row[axis]
                            else:
                                new_row['data'] = np.dstack((new_row['data'], new_row[axis]))
                        row_num = len(SpellerConstant.SENSOR_AXES) * data_len
                        new_row['data'] = np.ravel(new_row['data'], order='C')
                        new_row['target'] = labels[letter]
                        new_row['len'] = length
                        data_frame = data_frame.append(new_row, ignore_index=True)

    return data_frame
