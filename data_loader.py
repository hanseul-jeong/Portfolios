import pandas as pd
import numpy as np
import sqlite3
import os, sys
from utils import normalize

class data_loader():
    def __init__(self, StartDate=20100101, EndDate=20200101, data_dir='Dataset', data_='KOSPI_daily.db', selective=None):
        data_path = os.path.join(data_dir, data_)
        self.n_rawFeature = 6
        self.n_features = None
        self.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        # check directory
        if not os.path.exists(data_dir):
            print('Plz locate dataset file(db) in {dir} directory'.format(dir=data_dir))
            os.mkdir(data_dir)
            sys.exit()

        # dictinary from stock code to name (e.g., A00020 -> 동화약품 )
        df = pd.read_csv(os.path.join(data_dir,'code_dict.csv'))
        code_list = df['Code'].tolist()
        name_list = df['Name'].tolist()
        self.code_to_name = {code: name for code, name in zip(code_list, name_list)}
        self.name_to_code = {name: code for code, name in zip(code_list, name_list)}

        # load data
        rawData = []
        conn = sqlite3.connect(data_path)
        cursor = conn.cursor()
        n_samples = 0   # maximum sample numbers
        features = self.columns[0] + "".join([colon+col for colon, col in zip([', ']*(self.n_rawFeature-1), self.columns[1:])])

        # asset range
        if selective is not None:
            code_list = [self.name_to_code[name] for name in selective]

        for i, code in enumerate(code_list):
            sql = "Select {features} from {code} where date >= {StartDate} and date <= {EndDate}"\
                .format(features=features, code=code, StartDate=StartDate, EndDate=EndDate)
            cursor.execute(sql)
            d = cursor.fetchall()
            rawData.append(d)
            n_samples = len(d) if len(d) > n_samples else n_samples
        conn.close()

        exc_idx = []
        # exclude short history stock
        for i, d in enumerate(rawData):
            if len(d) != n_samples:
                exc_idx.append(i)

        n_assets = len(rawData) - len(exc_idx)
        self.Data = np.zeros([n_assets, n_samples, self.n_rawFeature], dtype=np.float32)
        self.code_list = []
        idx = 0
        for i in range(len(rawData)):
            if i not in exc_idx:
                self.Data[idx, :, :] = rawData[i]
                self.code_list.append(code_list[i])
                idx+= 1

    def select_col(self, type):

        if type == 'OHLC':
            choosen = ['open', 'high','low', 'close']
        elif type == 'OHLCV':
            choosen = ['open', 'high','low', 'close', 'volume']
        elif type == 'C':
            choosen = ['close']
        else:
            print('Plz select features')
            sys.exit()
        col_mask = np.array([True if c in choosen else False for c in self.columns])

        return choosen, col_mask

    def select_row(self, volumes, n_selected=100):
        '''
            select assets based on volume size
        :param volumes: averaged volume data. nd.array [A] (in training).
        :param n_selected: how many assets to select
        :return: bool mask. bool-type list [A] e.g., [True, False, ..., True]
        '''
        A = np.shape(volumes)[0]
        idx_max = np.argsort(-volumes)[:n_selected]
        row_mask = [True if idx in idx_max else False for idx in range(A)]

        return row_mask

    def load_data(self, type='OHLC', ValidDate=20150101, n_window=60, n_slide=1):
        '''

        :param type: Type of features 'OHLC', 'OHLCV,'V'
        :param ValidDate: A date when validation starts
        :param n_window: A size of sliding window
        :param n_slide: A step of sliding
        train, valid : 4-d ndarray [Assets, Times, Windows, Columns]
        train_x, valid_x : price ratio. 2-d ndarray [Assets, Times]
        :return: train, train_x, valid, valid_x
        '''

        choosen, col_mask = self.select_col(type)
        self.n_features = len(choosen)
        A, T, C = np.shape(self.Data)

        # ass_mask = self.selective()
        raw_data = np.array([self.Data[:, t:t+n_window, :] for t in range(0, T-n_window+1, n_slide)])   # T, A, W, C
        raw_data = raw_data.transpose([1, 0,2,3])   # A, T, W, C
        # x_t+1/x_t
        y = raw_data[:, 1:, 0, self.columns.index('close')]/raw_data[:, :-1, -1, self.columns.index('close')]
        n_train = (raw_data[0,:,-1,self.columns.index('date')] < ValidDate).sum()

        volumes = raw_data[:, :n_train, -1, self.columns.index('volume')].mean(1)
        row_mask = self.select_row(volumes)

        raw_data = raw_data[row_mask,:,:,:]
        y = y[row_mask, :]
        normed_data = normalize(raw_data, self.columns)
        train, valid = normed_data[:, :n_train, :, col_mask], normed_data[:, n_train:, :, col_mask]
        valid = valid[:, :-1]  # remove the last due to labels

        train_x, valid_x = y[:, :n_train], y[:, n_train:]
        return train, train_x, valid, valid_x
