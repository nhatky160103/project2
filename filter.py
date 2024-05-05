import tkinter as tk

import cv2
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageEnhance
from PIL import Image, ImageTk
from tkinter import ttk

import numpy as np

sobelx =np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
sobely=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])#trích xuất dặc trưng theo đường ngang
identity=np.array([[0,0,0],[0,1,0], [0,0,0]])

edge_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
sharpen=np.array([[0,-1,0],[-1,5,-1], [0,-1,0]])




def apply_custom_filter(image, kernel):

    # Tạo các ảnh kênh màu riêng lẻ
    b, g, r = cv2.split(image)

    # Áp dụng filter cho từng kênh màu
    b_filtered = cv2.filter2D(b, -1, kernel)

    g_filtered = cv2.filter2D(g, -1, kernel)

    r_filtered = cv2.filter2D(r, -1, kernel)

    # Kết hợp các kênh màu đã được xử lý thành ảnh màu hoàn chỉnh
    filtered_image = cv2.merge([b_filtered, g_filtered, r_filtered])
    filtered_image = Image.fromarray(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    return filtered_image




def view_tear_off_action():
    print("View menu has been torn off")


def apply_filter(filter, file_path,  canvas):

    print(file_path)
    image= Image.open(file_path)

    width, height = image.size
    # Tính toán tỷ lệ resize
    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height

    # Lấy tỷ lệ resize tốt nhất để không vượt quá kích thước của canvas
    resize_ratio = min(width_ratio, height_ratio)

    # Resize ảnh với tỷ lệ đã tính toán
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

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
        image= cv2.imread(file_path)
        image= apply_custom_filter(image, sobelx)
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
    image = image.resize((new_width,new_height))
    image = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.image = image
    canvas.create_image(x_offset, y_offset, image=image, anchor="nw")

    return result


def update_value(value,  file_path,  canvas):
    box_blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / (value+3))
    image = Image.open(file_path)

    width, height = image.size
    # Tính toán tỷ lệ resize
    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height

    # Lấy tỷ lệ resize tốt nhất để không vượt quá kích thước của canvas
    resize_ratio = min(width_ratio, height_ratio)

    # Resize ảnh với tỷ lệ đã tính toán
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2
    image = cv2.imread(file_path)
    image = apply_custom_filter(image, box_blur)
    result= image = image.resize((new_width, new_height))
    image = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.image = image
    canvas.create_image(x_offset, y_offset, image=image, anchor="nw")

    result = result.convert('RGB')
    result = np.array(result)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

import cv2
from PIL import Image, ImageTk

def edge_detection(file_path, canvas, value):
    image_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = image_grey.shape

    # Tính toán tỷ lệ resize
    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    resize_ratio = min(width_ratio, height_ratio)

    # Resize ảnh với tỷ lệ đã tính toán
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    # Thực hiện phát hiện biên và chuyển đổi sang đối tượng Image
    result = edges = cv2.Canny(image_grey, value*10, value*20)
    edges_image = Image.fromarray(edges)

    # Resize ảnh
    edges_image = edges_image.resize((new_width, new_height))

    # Chuyển đổi thành đối tượng ImageTk
    image = ImageTk.PhotoImage(edges_image)

    # Vị trí hiển thị trên canvas
    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

    # Xóa hình ảnh hiện tại và vẽ ảnh mới lên canvas
    canvas.delete("all")
    canvas.image = image
    canvas.create_image(x_offset, y_offset, image=image, anchor="nw")

    result= cv2.cvtColor(result,  cv2.COLOR_BGR2RGB)
    return result


def change_to_threshold(file_path, canvas ):

    image_grey=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = image_grey.shape

    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    resize_ratio = min(width_ratio, height_ratio)

    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    image_binary = np.zeros_like(image_grey)
    for i in range(image_grey.shape[0]):
        for j in range(image_grey.shape[1]):
            if image_grey[i][j] > 255 // 2:
                image_binary[i][j] = 255
            else:
                image_binary[i][j] = 0
    result= image_binary
    image_binary = Image.fromarray(image_binary)

    image_binary = image_binary.resize((new_width, new_height))

    image = ImageTk.PhotoImage(image_binary)

    # Vị trí hiển thị trên canvas
    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

    canvas.delete("all")
    canvas.image = image
    canvas.create_image(x_offset, y_offset, image=image, anchor="nw")

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def change_blur(file_path, canvas, sigma):
    image_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = image_grey.shape

    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    resize_ratio = min(width_ratio, height_ratio)

    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    result= loss_image = cv2.GaussianBlur(image_grey, ksize = (13, 13), sigmaX=sigma/4)
    loss_image = Image.fromarray(loss_image)

    loss_image = loss_image.resize((new_width, new_height))

    image = ImageTk.PhotoImage(loss_image)

    # Vị trí hiển thị trên canvas
    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

    canvas.delete("all")
    canvas.image = image
    canvas.create_image(x_offset, y_offset, image=image, anchor="nw")

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

def change_contrast(file_path,canvas, value):
    img = Image.open(file_path)
    width, height = img.size

    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    resize_ratio = min(width_ratio, height_ratio)

    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    # Enhance constrast
    enhancer = ImageEnhance.Contrast(img)
    factor = value / 10.0
    result=  new_img = enhancer.enhance(factor)

    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

    new_img = new_img.resize((new_width, new_height), Image.LANCZOS)

    canvas.canvas_image = ImageTk.PhotoImage(new_img)
    canvas.delete("all")
    canvas.create_image(x_offset, y_offset, image=canvas.canvas_image, anchor="nw")

    result = result.convert('RGB')
    result = np.array(result)
    result=  cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result