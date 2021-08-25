# GLOBAL LIBRARIES

import os
from collections import defaultdict
from pathlib import Path
from sys import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

# LOCAL PROJECT LIBRARIES
from strings import *
# CONFIG
from tools.data_generator import DataGenerator
from tools.metrics import dice_coef, preprocess

pd.set_option("display.max_rows", 101)
plt.rcParams["font.size"] = 15
train_df = pd.read_csv("input/train.csv")
sample_df = pd.read_csv("input/sample_submission.csv")
train_df.head()
print(f'{directory_structure}\n{os.listdir("input")}')
# END CONFIG


class_dict = defaultdict(int)

kind_class_dict = defaultdict(int)

no_defects_num = 0
defects_num = 0

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col + 4, 1]
    if labels.isna().all():
        no_defects_num += 1
    else:
        defects_num += 1

    kind_class_dict[sum(labels.isna().values == False)] += 1

    for idx, label in enumerate(labels.isna().values.tolist()):
        if label == False:
            class_dict[idx + 1] += 1

print("the number of images with no defects: {}".format(
    no_defects_num))  # the output should be "the number of images with no defects: N the number of images with defects: Nn"
print("the number of images with defects: {}".format(defects_num))

fig, ax = plt.subplots()
sns.barplot(x=list(class_dict.keys()), y=list(class_dict.values()), ax=ax)
ax.set_title("the number of images for each class")
ax.set_xlabel("class")
print(class_dict)

fig, ax = plt.subplots()
sns.barplot(x=list(kind_class_dict.keys()), y=list(kind_class_dict.values()), ax=ax)
ax.set_title("Number of classes included in each image");
ax.set_xlabel("number of classes in the image")
print(kind_class_dict)

train_size_dict = defaultdict(int)
train_path = Path("input/train_images/")

for img_name in train_path.iterdir():
    img = Image.open(img_name)
    train_size_dict[img.size] += 1
print(train_size_dict)

test_size_dict = defaultdict(int)
test_path = Path("input/test_images/")

for img_name in test_path.iterdir():
    img = Image.open(img_name)
    test_size_dict[img.size] += 1

print(test_size_dict)

from tools.metrics import model

# SAVE MODEL
model.save('UNET.h5')

# LOAD MODEL
from keras.models import load_model

model = load_model('UNET.h5', custom_objects={'dice_coef': dice_coef})

# PREDICT 1 BATCH TEST DATASET
test = pd.read_csv(path + 'sample_submission.csv')
test['ImageId'] = test['ImageId_ClassId'].map(lambda x: x.split('_')[0])
test_batches = DataGenerator(test.iloc[::4], subset='test', batch_size=256, preprocess=preprocess)
test_preds = model.predict_generator(test_batches, steps=1, verbose=1)
