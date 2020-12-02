import pandas as pd
import os
import sqlite3

data_path = os.path.join('Data', 'KOSPI_daily.db')
dict_path = os.path.join('Data','code_dict.csv')
df = pd.read_csv(dict_path, encoding='utf-8-sig')
code_to_name = {code:name for code, name in zip(df['Code'], df['Name'])}

conn = sqlite3.connect(data_path)
cursor = conn.cursor()

dates = []

for code in df['Code']:
    sql = 'SELECT date FROM {code}'.format(code=code)
    cursor.execute(sql)
    dates.append(cursor.fetchall()[0][0])

cursor.close()
