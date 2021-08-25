import tensorflow as tf
from _pytest.monkeypatch import K
from keras.losses import binary_crossentropy
from matplotlib import pyplot as plt

from tools.data_generator import DataGenerator
from tools.unet import train2


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# LOSS FUNCTION

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet
def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = tf.keras.layers.Flatten()(y_true)
    y_pred_pos = tf.keras.layers.Flatten()(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1 - pt_1), gamma)


from segmentation_models import Unet
from segmentation_models import get_preprocessing

# LOAD UNET WITH PRETRAINING FROM IMAGENET
preprocess = get_preprocessing('resnet34')  # for resnet, img = (img-110.0)/1.0
model = Unet('resnet34', input_shape=(128, 800, 3), classes=4, activation='sigmoid')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
# adam = tf.keras.optimizers.Adam(lr = 0.05, epsilon = 0.1)
# model.complie(optimizer = adam, loss = focal_tversky_loss, metrics = [tversky, dice_coef])

# TRAIN AND VALIDATE MODEL
idx = int(0.8 * len(train2));
print()
train_batches = DataGenerator(train2.iloc[:idx], shuffle=True, preprocess=preprocess)
valid_batches = DataGenerator(train2.iloc[idx:], preprocess=preprocess)
history = model.fit_generator(train_batches, validation_data=valid_batches, epochs=30, verbose=2)

# PLOT TRAINING
plt.figure(figsize=(15, 5))
plt.plot(range(history.epoch[-1] + 1), history.history['val_dice_coef'], label='val_dice_coef')
plt.plot(range(history.epoch[-1] + 1), history.history['dice_coef'], label='trn_dice_coef')
plt.title('Training Accuracy');
plt.xlabel('Epoch');
plt.ylabel('Dice_coef');
plt.legend();
plt.show()
