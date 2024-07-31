# MODEL ĐƯỢC TRAIN TRÊN KAGGLE NÊN ĐƯỜNG DẪN FILE DÚNG VỚI ĐƯỜNG DẪN NOTEBOOK TRÊN KAGGLE
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from load_data import create_dir
H = 512
W = 512

smooth = 1e-15

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Các hàm và biến cần thiết kết thúc

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Directory for storing files """
create_dir("test_images/mask")

""" Loading model """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model("/kaggle/input/model-human-segmentation/model_human_segmentation.h5")



import matplotlib.pyplot as plt
path="/kaggle/input/test-image-segmentation/test_image.jpg"
path= "/kaggle/input/test-image-segmentation"
file_path= sorted(glob(os.path.join(path,  "*.jpg")))


for path in file_path:

    plt.figure(figsize=(20, 20))


    image = cv2.imread(path, cv2.IMREAD_COLOR)

    h, w, _ = image.shape
    x = cv2.resize(image, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Prediction """
    y = model.predict(x)[0]
    y = cv2.resize(y, (w, h))


    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')

    # Hiển thị ảnh thứ hai trong cột 2
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(y, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')

    # Hiển thị biểu đồ
    plt.show()