import cv2, io
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
LIST_CHAR = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
fix_param=[(0,0), (0,0), (2,1.5), (2,2), (3,2), (4,2), (5,1.5), (5,2)]





def create_ascii_image(file_path, canvas, char_size, mode):
    image = cv2.imread(file_path)
    height, width,_ = image.shape

    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height

    # Lấy tỷ lệ resize tốt nhất để không vượt quá kích thước của canvas
    resize_ratio = min(width_ratio, height_ratio)

    # Resize ảnh với tỷ lệ đã tính toán
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    image = cv2.resize(image, (new_width, new_height))
    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

    num_cols = new_width // fix_param[char_size][0] # Điều chỉnh số cột dựa trên chiều rộng của canvas
    cell_width = new_width // num_cols
    cell_height =fix_param[char_size][1]*cell_width
    num_rows = int(new_height // cell_height)

    font = ImageFont.load_default()

    char_width, char_height= font.getsize("A")
    color_image = Image.new("RGB", (num_cols*char_width, num_rows*char_height), (0, 0, 0))
    draw2 = ImageDraw.Draw(color_image)
    non_color_image = Image.new("RGB", (num_cols*char_width, num_rows*char_height), (255, 255, 255))
    draw = ImageDraw.Draw(non_color_image)
    for i in range(num_rows):
        for j in range(num_cols):
            # non color
            sub_image = image[int(i * cell_height): int((i + 1) * cell_height),
                        int(j * cell_width): int((j + 1) * cell_width)]
            index = int((np.mean(sub_image) / 255) * (len(LIST_CHAR) - 1))

            #color
            avg_color= np.sum(np.sum(sub_image, axis=0),axis=0 )/(cell_height*cell_width)
            avg_color= tuple(avg_color.astype(np.int32).tolist())
            draw2.text((j*char_width,i* char_height), LIST_CHAR[index], fill= avg_color, font= font)
            draw.text((j * char_width, i * char_height), LIST_CHAR[index], fill="black", font=font)

    if mode==1:
        display_image= color_image
    else:
        display_image = non_color_image

    resized_image = display_image.resize((new_width, new_height), Image.LANCZOS)
    canvas_image = ImageTk.PhotoImage(resized_image)
    canvas.delete("all")
    canvas.image= canvas_image
    canvas.create_image(x_offset, y_offset, image=canvas_image, anchor="nw")




    non_color_image = np.array(non_color_image)
    color_image = np.array(color_image)


    if mode==1:
        return color_image
    else:
        return non_color_image



