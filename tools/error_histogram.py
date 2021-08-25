import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from neural import idx
from tools.error_detect import preds
from tools.unet import train2

pix_min = 250
for THRESHOLD in [0.1, 0.25, 0.50, 0.75, 0.9]:
    print('######################################')
    print('## Threshold =', THRESHOLD, 'displayed below ##')
    print('######################################')
    correct = [[], [], [], []];
    incorrect = [[], [], [], []]
    for i, f in enumerate(train2.iloc[idx:idx + len(preds)]['ImageId']):
        preds2 = preds[i].copy()
        preds2[preds2 >= THRESHOLD] = 1
        preds2[preds2 < THRESHOLD] = 0
        sums = np.sum(preds2, axis=(0, 1))
        for j in range(4):
            if 4 * sums[j] < pix_min: continue
            if train2.iloc[i, j + 1] == '':
                incorrect[j].append(4 * sums[j])
            else:
                correct[j].append(4 * sums[j])
    plt.figure(figsize=(20, 8))
    for j in range(4):
        limit = [10000, 10000, 100000, 100000][j]
        plt.subplot(2, 2, j + 1)
        sns.distplot([x for x in correct[j] if x < limit], label='correct')
        sns.distplot([x for x in incorrect[j] if x < limit], label='incorrect')
        plt.title('Defect ' + str(j + 1) + ' mask sizes with threshold = ' + str(THRESHOLD));
        plt.legend()
    plt.show()
    for j in range(4):
        c1 = np.array(correct[j])
        c2 = np.array(incorrect[j])
        print('With threshold =', THRESHOLD, ', defect', j + 1, 'has', len(c1[c1 != 0]), 'correct and',
              len(c2[c2 != 0]), 'incorrect masks')
    print()
