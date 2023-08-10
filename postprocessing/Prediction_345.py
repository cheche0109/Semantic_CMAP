import pandas as pd
import numpy as np
import os

for filename in os.listdir('./Projection_BF'):
    print(filename)
    enfor_nn = []
    with open('./Projection_BF/'+filename, 'r') as f:
        for line in f:
            enfor_nn.append((line.strip().split(' ')[1]))
    with open('./Prediction_E/' + filename, 'r') as f:
      for line in f:
        original_lst = line.strip().split(' ')
        id = original_lst[0]
        idx = original_lst[1]
        ss = original_lst[2]
        score = float(original_lst[-1])
        pos = ' ' + idx + ' '
        if idx in enfor_nn:
          file = open('./Projection_BF/' + filename,mode='r')
          enfor_file = file.read()
          enfor_line = [line for line in enfor_file.split('\n') if pos in line]
          enfor_lst = enfor_line[0].split(' ')
          ss = 'E'
          with open('./Prediction_45/'+filename, 'a') as fw:
             fw.write(id + ' ' + idx + ' ' + ss + ' ' + str(enfor_lst[-1]) + '\n' )
        else:
          with open('./Prediction_45/'+filename, 'a') as fw:
             fw.write(id + ' ' + idx + ' ' + ss + ' ' + str(score) + '\n' )
