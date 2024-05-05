import cv2
import numpy as np
import glob
import tkinter as tk
from PIL import Image, ImageTk

def get_image(path, cell_size):
    result = []
    images = []
    image_paths = glob.glob("{}/*.jpg".format(path))
    for image_path in image_paths:
        sub_img = cv2.imread(image_path)
        sub_img = cv2.resize(sub_img, (cell_size, cell_size), interpolation=cv2.INTER_CUBIC)
        avg_color = np.mean(sub_img, axis=(0, 1))
        result.append(avg_color)
        images.append(sub_img)
    return result, images

def create_photomosaic(file_path, canvas, cell_size):
    image_input = cv2.imread(file_path)
    height, width, _ = image_input.shape

    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    # Lấy tỷ lệ resize tốt nhất để không vượt quá kích thước của canvas
    resize_ratio = min(width_ratio, height_ratio)

    # Resize ảnh với tỷ lệ đã tính toán
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    # image_input = cv2.resize(image_input, (new_width, new_height))
    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2


    num_cols = width // cell_size
    num_rows = height // cell_size

    sub_arr, pool_image = get_image("D:/java_workspace/face_segmentation/lapa/sub_image", cell_size)
    output_image = np.zeros((height, width, 3))
    for i in range(num_cols):
        for j in range(num_rows):
            sub_image = image_input[j * cell_size: (j + 1) * cell_size, i * cell_size: (i + 1) * cell_size]
            sub_avg_color = np.mean(sub_image, axis=(0, 1))
            index = np.argmin(np.sqrt(np.sum((sub_avg_color - sub_arr) ** 2, axis=1)))
            output_image[j * cell_size: (j + 1) * cell_size, i * cell_size: (i + 1) * cell_size] = pool_image[index]

    canvas.delete("all")
    output_image= cv2.resize(output_image,(new_width, new_height))
    result= output_image_normalized = cv2.normalize(output_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U)
    output_image_normalized = Image.fromarray(output_image_normalized)
    output_image_tk = ImageTk.PhotoImage(output_image_normalized)
    canvas.image = output_image_tk
    canvas.create_image(x_offset, y_offset, image=output_image_tk, anchor="nw")
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

