# PREDICT FROM VALIDATION SET (ONLY IMAGES WITH DEFECTS 1, 2, 4)
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from neural import idx
from tools.data_generator import DataGenerator, mask2pad, mask2contour
from tools.metrics import preprocess, model
from tools.unet import train2

val_set = train2.iloc[idx:]
val_set2 = val_set[(val_set['count'] != 0) & (val_set['e3'] == '')].sample(16)

valid_batches = DataGenerator(val_set2, preprocess=preprocess)
preds = model.predict_generator(valid_batches, verbose=1)

# PLOT PREDICTIONS
valid_batches = DataGenerator(val_set2)
print('Plotting predictions...')
print('KEY: yellow=defect1, green=defect2, blue=defect3, magenta=defect4')
for i, batch in enumerate(valid_batches):
    plt.figure(figsize=(20, 36))
    for k in range(16):
        plt.subplot(16, 2, 2 * k + 1)
        img = batch[0][k,]
        img = Image.fromarray(img.astype('uint8'))
        img = np.array(img)
        dft = 0
        three = False
        for j in range(4):
            msk = batch[1][k, :, :, j]
            if (j == 2) & (np.sum(msk) != 0):
                three = np.sum(msk)
            msk = mask2pad(msk, pad=2)
            msk = mask2contour(msk, width=3)
            if j == 0:  # yellow
                img[msk == 1, 0] = 235
                img[msk == 1, 1] = 235
            elif j == 1:
                img[msk == 1, 1] = 210  # green
            elif j == 2:
                img[msk == 1, 2] = 255  # blue
            elif j == 3:  # magenta
                img[msk == 1, 0] = 255
                img[msk == 1, 2] = 255
        extra = '';
        extra2 = ''
        if not three:
            extra = 'NO DEFECT 3'
            extra2 = 'ERROR '
        plt.title('Train ' + train2.iloc[16 * i + k, 0] + '  ' + extra)
        plt.axis('off')
        plt.imshow(img)
        plt.subplot(16, 2, 2 * k + 2)
        dft = 3
        if dft != 0:
            msk = preds[16 * i + k, :, :, dft - 1]
            plt.imshow(msk)
        else:
            plt.imshow(np.zeros((128, 800)))
        plt.axis('off')
        mx = np.round(np.max(msk), 3)
        plt.title(extra2 + 'Predict Defect ' + str(dft) + '  (max pixel = ' + str(mx) + ')')
    plt.subplots_adjust(wspace=0.05)
    plt.show()
