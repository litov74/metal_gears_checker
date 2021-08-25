import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

path = 'input/'
train = pd.read_csv(path + 'train.csv')

# RESTRUCTURE TRAIN DATAFRAME
train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0] + '.jpg')
train2 = pd.DataFrame({'ImageId': train['ImageId'][::4]})
train2['e1'] = train['EncodedPixels'][::4].values
train2['e2'] = train['EncodedPixels'][1::4].values
train2['e3'] = train['EncodedPixels'][2::4].values
train2['e4'] = train['EncodedPixels'][3::4].values
train2.reset_index(inplace=True, drop=True)
train2.fillna('', inplace=True);
train2['count'] = np.sum(train2.iloc[:, 1:] != '', axis=1).values
train2.head()
