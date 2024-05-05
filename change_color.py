import cv2
import numpy as np
import skimage.exposure
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import ImageTk, Image
import tensorflow as tf


label_dict={
	"background": 0,
	"skin": 1,
	"eyebrow": 2,
	"eye":4,
	"nose":6,
	"lip":7,
	"inner mouth":8,
	"hair":10,
}

beta={
	"background": 0,
	"skin": 0.1,
	"eyebrow": 0.2,
	"eye":0.6,
	"nose":0.1,
	"lip":0.7,
	"inner mouth":0.3,
	"hair":0.7,
}

def find_mask(image, model, num_label):

    # image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    height, width, dim = image.shape
    image = cv2.resize(image, (512, 512))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    y_predict = model.predict(image, verbose=0)[0]
    y_predict = np.argmax(y_predict, axis=-1)
    y_predict = y_predict.astype(np.uint8)


    mask = convert_to_mask(y_predict, num_label)
    mask = cv2.resize(mask, (width, height))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#[[0,1,0][1,1,1][0,1,0]]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def convert_to_mask(y, num_label):
    if num_label==0 or num_label==1 or num_label==6 or num_label==8 or num_label==10 :
        for i in range(512):
            for j in range(512):
                if y[i][j] == num_label:
                    y[i][j]=255
                else:
                    y[i][j]=0
        return y.astype(np.uint8)
    elif num_label==2:
        for i in range(512):
            for j in range(512):
                if y[i][j] == 2 or y[i][j] == 3:
                    y[i][j]=255
                else:
                    y[i][j]=0
        return y.astype(np.uint8)
    elif num_label==4:
        for i in range(512):
            for j in range(512):
                if y[i][j] == 4 or y[i][j] == 5:
                    y[i][j]=255
                else:
                    y[i][j]=0
        return y.astype(np.uint8)
    elif num_label==7:
        for i in range(512):
            for j in range(512):
                if y[i][j] == 7 or y[i][j] == 9:
                    y[i][j]=255
                else:
                    y[i][j]=0
        return y.astype(np.uint8)

def change_color_face( canvas, img, num_label, color):
    tem_beta= beta[num_label]
    num_label = label_dict[num_label]

    model = tf.keras.models.load_model("models/model_29_3.h5")

    mask= find_mask(img, model, num_label)

    # specify desired bgr color
    desired_color = color[::-1]
    desired_color = np.asarray(desired_color, dtype=np.float64)
    print(desired_color)


    swatch = np.full((200,200,3), desired_color, dtype=np.uint8)

    # get average bgr color of mask
    avg_color = cv2.mean(img, mask=mask)[:3]
    print("avg_color",avg_color)
    print("desired color", desired_color)


    # compute difference colors and make into an image the same size as input
    diff_color = desired_color - avg_color
    print("diff_color",diff_color)
    diff_color = np.full_like(img, diff_color, dtype=np.int32)

    img2= img.astype(np.int32)

    new_img = cv2.addWeighted(img2, 1, diff_color, tem_beta, 0)
    new_img = new_img.clip(0, 255).astype(np.uint8)

    #antialias mask, convert to float in range 0 to 1 and make 3-channels
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
    mask = skimage.exposure.rescale_intensity(mask, in_range=(128,255), out_range=(0,1)).astype(np.float32)
    mask = cv2.merge([mask,mask,mask])

    # combine img and new_img using mask
    result = (img * (1 - mask) + new_img * mask)
    result = result.clip(0,255).astype(np.uint8)




    #Hien thi hinh anh len canvas
    height, width,_ = img.shape

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



def main():
    result= change_color_face(1, skin_rgb[3])
    cv2.imshow("image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()