import os
import pandas as pd
import numpy as np
for filename in os.listdir('./Projection_bottom_final'):
    print(filename)
    pred = pd.read_csv('./Projection_bottom_final/' + filename,  sep=" ", names = ['id', 'n', 'ss', 'score'])
    table_sorted = pred.sort_values(by=['n', 'score'], ascending=(True, False))
    table = table_sorted.drop_duplicates("n", keep='first').reset_index(drop=True)
    table.to_csv(r'./Projection_BF/'+filename, header=None, index=None, sep=' ', mode='a')
