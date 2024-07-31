# MODEL ĐƯỢC TRAIN TRÊN KAGGLE NÊN ĐƯỜNG DẪN FILE DÚNG VỚI ĐƯỜNG DẪN NOTEBOOK TRÊN KAGGLE
import tensorflow as tf
import os
from glob import glob
from sklearn.model_selection import train_test_split
import numpy  as np
from data_augment import augment_data


print ( tf.__version__)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# !rm -rf /kaggle/working/*


def load_data(path, split=0.1):
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    for x, y in zip(X, Y):
#         print(x, y)
        split_size= int(split*len(X))
        train_x, test_x= train_test_split(X, test_size=split_size, random_state=42)
        train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)
        return (train_x, train_y),(test_x, test_y)



""" creating a directory"""
np.random.seed(42)


data_path = "/kaggle/input/human-segmentation-data-set/people_segmentation"
(train_x, train_y),(test_x, test_y) = load_data(data_path)
print("Train", len(train_x), len(train_y))
print("Test", len(test_x), len(test_y))

""" Create directories to save the augmented data """
create_dir("/kaggle/working/new_data/train/image/")
create_dir("/kaggle/working/new_data/train/mask/")
create_dir("/kaggle/working/new_data/test/image/")
create_dir("/kaggle/working/new_data/test/mask/")



# """ Data augmentation """
# augment_data(train_x, train_y, "/kaggle/working/new_data/train", augment=True)
augment_data(test_x, test_y, "/kaggle/working/new_data/test", augment= False)

