import tkinter as tk
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageEnhance
import cv2
from PIL import Image, ImageTk
import numpy as np


sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

edge_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def apply_custom_filter(image, kernel):

    b, g, r = cv2.split(image)

    # Áp dụng filter cho từng kênh màu
    b_filtered = cv2.filter2D(b, -1, kernel)

    g_filtered = cv2.filter2D(g, -1, kernel)

    r_filtered = cv2.filter2D(r, -1, kernel)

    # Kết hợp các kênh màu đã được xử lý thành ảnh màu hoàn chỉnh
    filtered_image = cv2.merge([b_filtered, g_filtered, r_filtered])
    filtered_image = Image.fromarray(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    return filtered_image


def show_image(height, width, image,  canvas):
    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    resize_ratio = min(width_ratio, height_ratio)
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2
    image = image.resize((new_width, new_height))
    image = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.image = image
    canvas.create_image(x_offset, y_offset, image=image, anchor="nw")


def apply_filter(filter, file_path,  canvas):

    image = Image.open(file_path)
    width, height = image.size


    if filter == "Black and White":
        image = ImageOps.grayscale(image)
    elif filter == "Blur":
        image = image.filter(ImageFilter.BLUR)
    elif filter == "Sharpen":
        image = image.filter(ImageFilter.SHARPEN)
    elif filter == "Smooth":
        image = image.filter(ImageFilter.SMOOTH)
    elif filter == "Emboss":
        image = image.filter(ImageFilter.EMBOSS)

    elif filter == "sobel_x":
        image = cv2.imread(file_path)
        image = apply_custom_filter(image, sobelx)
    elif filter == "sobel_y":
        image = cv2.imread(file_path)
        image = apply_custom_filter(image, sobely)
    elif filter == "identity":
        image = cv2.imread(file_path)
        image = apply_custom_filter(image, identity)
    elif filter == "edge_detection":
        image = cv2.imread(file_path)
        image = apply_custom_filter(image, edge_detection_kernel)
    elif filter == "sharpen":
        image = cv2.imread(file_path)
        image = apply_custom_filter(image, sharpen)
    elif filter == "vignette":
        image = cv2.imread(file_path)
        rows, cols = image.shape[:2]

        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)

        for i in range(3):
            image[:, :, i] = image[:, :, i] * mask
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image_rgb = image.convert('RGB')
    result = np.array(image_rgb)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    show_image(height, width, image,  canvas)


    return result



def change_light(value,  file_path,  canvas):
    box_blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / (value+3))

    image = cv2.imread(file_path)
    height,width,_ = image.shape

    image = apply_custom_filter(image, box_blur)
    result = image
    result = result.convert('RGB')
    result = np.array(result)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    show_image(height, width, image, canvas)

    return result




def edge_change(file_path, canvas, value):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape

    result = edges = cv2.Canny(image, value*10, value*20)
    image = Image.fromarray(edges)

    show_image(height, width, image, canvas)

    result = cv2.cvtColor(result,  cv2.COLOR_BGR2RGB)

    return result


def change_to_threshold(file_path, canvas ):

    image_grey=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = image_grey.shape

    image_binary = np.zeros_like(image_grey)
    for i in range(image_grey.shape[0]):
        for j in range(image_grey.shape[1]):
            if image_grey[i][j] > 255 // 2:
                image_binary[i][j] = 255
            else:
                image_binary[i][j] = 0
    result = image_binary
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_binary)

    show_image(height, width, image, canvas)

    return result


def change_blur(file_path, canvas, sigma):
    image_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = image_grey.shape

    result = loss_image = cv2.GaussianBlur(image_grey, ksize = (13, 13), sigmaX=sigma/4)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(loss_image)

    show_image(height, width, image, canvas)
    return result


def change_contrast(file_path,canvas, value):
    img = Image.open(file_path)
    width, height = img.size

    # Enhance constrast
    enhancer = ImageEnhance.Contrast(img)
    factor = value / 10.0
    result = image = enhancer.enhance(factor)
    result = result.convert('RGB')
    result = np.array(result)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    show_image(height, width, image, canvas)

    return result