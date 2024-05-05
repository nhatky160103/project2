import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score
from train import create_dir, load_dataset

global image_h
global image_w
global num_classes
global classes
global rgb_codes



def convert_to_rgb_image(mask, rgb_code):
    h, w = mask.shape[0],  mask.shape[1]
    output=[]

    mask= mask.astype(np.int32)
    for i, pixel in enumerate(mask.flatten()):
        output.append(rgb_code[pixel])
    output= np.reshape(output, (h,w,3))
    return output

def save_result(image_x, mask, y_predict, save_image_path):
    mask= convert_to_rgb_image(mask, rgb_codes)
    y_predict= convert_to_rgb_image(y_predict, rgb_codes)
    line= np.ones((mask.shape[0], 10, 3 ))*255
    cat_images=np.concatenate([image_x, line, mask, line, y_predict], axis=1)
    cv2.imwrite(save_image_path, cat_images)
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Hyperparameters """
    image_h = 512
    image_w = 512
    num_classes = 11

    """ Paths """
    dataset_path = "D:/java_workspace/face_segmentation/lapa/LaPa"
    model_path = os.path.join("models", "model_29_3.h5")

    """ RGB Code and Classes """
    rgb_codes = [
        [0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
        [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
        [102, 0, 51], [255, 204, 255], [255, 0, 102]
    ]

    classes = [
        "background", "skin", "left eyebrow", "right eyebrow",
        "left eye", "right eye", "nose", "upper lip", "inner mouth",
        "lower lip", "hair"
    ]

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    test_x= test_x[:30]
    test_y= test_y[:30]
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Load the model """
    model= tf.keras.models.load_model(model_path)
    # model.summary()

    """prediction and evaluation"""
    SCORE=[]

    i=0;
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = x.split('/')[-1].split("\\")[-1].split('.')[0]
        print(name)
        i+=1;
        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image_w, image_h))
        image_x = image
        image = image / 255.0  ## (H, W, 3)
        image = np.expand_dims(image, axis=0)  ## [1, H, W, 3]
        image = image.astype(np.float32)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_w, image_h))
        mask = mask.astype(np.int32)


        y_predict= model.predict(image, verbose=0)[0]
        y_predict=np.argmax(y_predict, axis=-1)
        y_predict= y_predict.astype(np.int32)



        save_image_path = f"results/{name}.png"
        save_result(image_x, mask, y_predict, save_image_path)

        mask= mask.flatten()
        y_predict= y_predict.flatten()

        labels = [i for i in range(num_classes)]

        f1_value= f1_score(mask, y_predict,labels= labels, average=None, zero_division=0)
        jac_value = jaccard_score(mask, y_predict, labels=labels, average=None, zero_division=0)
        """save the result"""

        SCORE.append([f1_value, jac_value])

    score = np.array(SCORE)
    score= np.mean(score, axis=0)
    print(score)
    print(score.shape)

    f = open("files/score.csv", "w")
    f.write("Class,F1,Jaccard\n")


    l = ["Class", "F1", "Jaccard"]
    print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
    print("-" * 35)

    for i in range(num_classes):
        class_name = classes[i]
        f1 = score[0, i]
        jac = score[1, i]
        dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
        print(dstr)
        f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    print("-" * 35)
    class_mean = np.mean(score, axis=-1)
    class_name = "Mean"

    f1 = class_mean[0]
    jac = class_mean[1]

    dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    print(dstr)
    f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    f.close()


