import cv2
import numpy as np
import skimage.exposure
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import ImageTk, Image
import tensorflow as tf



beta = {
    "background": 0,
    "skin": 0.1,
    "eyebrow": 0.2,
    "eye": 0.6,
    "nose": 0.1,
    "lip": 0.7,
    "inner mouth": 0.3,
    "hair": 0.7,
}




def change_color_face(canvas, img, num_label, color, mask):
    tem_beta = beta[num_label]
    # specify desired bgr color
    desired_color = color[::-1]
    desired_color = np.asarray(desired_color, dtype=np.float64)
    print(desired_color)

    swatch = np.full((200, 200, 3), desired_color, dtype=np.uint8)

    # get average bgr color of mask
    avg_color = cv2.mean(img, mask=mask)[:3]
    print("avg_color", avg_color)
    print("desired color", desired_color)

    # compute difference colors and make into an image the same size as input
    diff_color = desired_color - avg_color
    print("diff_color", diff_color)
    diff_color = np.full_like(img, diff_color, dtype=np.int32)

    img2 = img.astype(np.int32)

    new_img = cv2.addWeighted(img2, 1, diff_color, tem_beta, 0)
    new_img = new_img.clip(0, 255).astype(np.uint8)

    # antialias mask, convert to float in range 0 to 1 and make 3-channels
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
    mask = skimage.exposure.rescale_intensity(mask, in_range=(128, 255), out_range=(0, 1)).astype(np.float32)
    mask = cv2.merge([mask, mask, mask])

    # combine img and new_img using mask
    result = (img * (1 - mask) + new_img * mask)
    result = result.clip(0, 255).astype(np.uint8)

    # Hien thi hinh anh len canvas
    height, width, _ = img.shape

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

    image_pil = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_pil)
    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)

    image_tk = ImageTk.PhotoImage(image_pil)

    canvas.delete("all")
    canvas.create_image(x_offset, y_offset, anchor="nw", image=image_tk)

    canvas.image = image_tk
    return result

