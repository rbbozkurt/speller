from firebase import firebase
import pandas as pd
import json
import SpellerConstant
import numpy as np


class SpellerDatabase:

    def __init__(self, firebase_ref, sensor_axes: []):
        self.fb_app = firebase.FirebaseApplication(firebase_ref, None)
        self.devices_db = self.fb_app.get(SpellerConstant.DEVICES_NODE_REFERENCE, None)

    # get all connected devices to database
    def get_connected_devices(self) -> []:
        connected_devices_id = []
        for device in self.devices_db.values():
            device_dic = json.loads(device)
            connected_devices_id.append(device_dic['id'])
        return connected_devices_id

    def read_letters_from_database(self, target_letters: []) -> pd.DataFrame():
        data_frame = pd.DataFrame()
        connected_devices_id = self.get_connected_devices()
        for device_id in connected_devices_id:
            path = '/{}'.format(device_id)
            # print(path)
            device_letters = self.fb_app.get(path, None)
            if (device_letters != None):
                # print("Result is not None:\nPrinting letters:\n{}".format(result.keys()))
                labels = {}
                # for num, letter in enumerate(result.keys(), start=1):
                for num, letter in enumerate(target_letters, start=1):

                    labels[letter] = float(num)
                    letter_path = '{}/{}'.format(path, letter)
                    record_date = self.fb_app.get(letter_path, None)
                    if (record_date != None):
                        for date in record_date.keys():
                            new_row = {'date': date}
                            record_path = '{}/{}'.format(letter_path, date)
                            # print(record_path)
                            data = self.fb_app.get(record_path, None)
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
