import cv2
import os
from PIL import ImageTk, Image
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

H = 512
W = 512

def change_human_background(canvas, file_path, background_img):
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("models/human_segmentaiton.h5")
    img = cv2.imread(file_path)
    height, width, _ = img.shape

    # Resize and normalize the background image
    back_img = cv2.resize(background_img, (width, height))
    back_img = back_img / 255.0
    back_img = back_img.astype(np.float32)

    # Resize and normalize the input image
    x = cv2.resize(img, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # Prediction
    y = model.predict(x)[0]
    y = cv2.resize(y, (width, height))
    y = np.expand_dims(y, axis=-1)

    # Invert the mask
    y2 = 1 - y

    # Apply the mask to the background image
    back_img = back_img * y2

    # Apply the mask to the input image
    result = img / 255.0 * y
    result = np.clip(result.astype(np.float32) + back_img, 0, 1) * 255.0
    result = result.astype(np.uint8)


    # Resize the result image to fit the canvas
    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    resize_ratio = min(width_ratio, height_ratio)
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

    # Convert the result image to PIL format and resize it
    image_pil = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_pil)
    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)

    # Convert the PIL image to ImageTk format and display it on the canvas
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.delete("all")
    canvas.create_image(x_offset, y_offset, anchor="nw", image=image_tk)
    canvas.image = image_tk

    return result

