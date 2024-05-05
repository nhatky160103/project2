
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
def create_rotate(file_path, canvas, angle):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(file_path)

    height, width, dim = image.shape

    width_ratio = canvas.winfo_width() / width
    height_ratio = canvas.winfo_height() / height
    resize_ratio = min(width_ratio, height_ratio)

    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    image= cv2.resize(image,(new_width, new_height))

    x_offset = (canvas.winfo_width() - new_width) // 2
    y_offset = (canvas.winfo_height() - new_height) // 2

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), angle, 1)
    result= rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))


    rotated_image = Image.fromarray(rotated_image)
    image = ImageTk.PhotoImage(rotated_image)


    # Xóa tất cả các đối tượng trên canvas và vẽ ảnh mới
    canvas.delete("all")
    canvas.image = image
    canvas.create_image(x_offset, y_offset, image=image, anchor="nw")
    result= cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

class ImageCutter:
    def __init__(self, file_path, canvas):
        self.file_path = file_path
        self.canvas = canvas
        self.canvas.config(cursor="cross")
        # Load ảnh gốc
        self.original_image = cv2.imread(self.file_path)

        height, width, dim = self.original_image.shape


        width_ratio = canvas.winfo_width() / width
        height_ratio = canvas.winfo_height() / height
        resize_ratio = min(width_ratio, height_ratio)

        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)
        self.original_image = cv2.resize(self.original_image, (new_width,new_height))

        self.x_offset = (canvas.winfo_width() - new_width) // 2
        self.y_offset = (canvas.winfo_height() - new_height) // 2


        # Tạo các biến để lưu giữ tọa độ của điểm bắt đầu và kết thúc khi chọn vùng cắt
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        # Gắn các sự kiện chuột
        self.canvas.bind("<ButtonPress-1>", self.on_click_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_click_end)

    def on_click_start(self, event):
        # Lưu tọa độ bắt đầu khi click chuột
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def on_drag(self, event):
        # Cập nhật tọa độ kết thúc khi kéo chuột
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)
        # Xóa các hình chữ nhật đã vẽ trước đó
        self.canvas.delete("rect")
        # Vẽ hình chữ nhật mới
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="rect")

    def on_click_end(self, event):
        # Hiển thị hộp thoại để nhập tên tệp mới
        new_file_name = filedialog.asksaveasfilename(defaultextension=".jpg")
        if new_file_name:
            # Cắt và lưu ảnh
            self.cut_and_save(new_file_name)


    def cut_and_save(self, new_file_name):
        if self.original_image is not None:
            # Lấy tọa độ góc trên bên trái và dưới bên phải của vùng cắt
            x1 = int(min(self.start_x, self.end_x))- self.x_offset
            y1 = int(min(self.start_y, self.end_y))-self.y_offset
            x2 = int(max(self.start_x, self.end_x))-self.x_offset
            y2 = int(max(self.start_y, self.end_y))-self.y_offset


            cropped_image = self.original_image[y1:y2, x1:x2]
            # Lưu ảnh cắt
            cv2.imwrite(new_file_name, cropped_image)
            messagebox.showinfo("Success", "Ảnh đã được cắt và lưu thành công!")
            print("Ảnh đã được cắt và lưu thành công!")


def zoom_image(image_path, canvas):

    image = Image.open(image_path)
    image_width, image_height = image.size


    x_offset = canvas.winfo_width()//2
    y_offset = canvas.winfo_height()// 2
    tk_image = ImageTk.PhotoImage(image)

    image_on_canvas = canvas.create_image(x_offset, y_offset, anchor="center", image=tk_image)

    zoom_factor = 1.0
    zoom_step = 0.1
    start_x = None
    start_y = None

    def zoom(event):
        nonlocal zoom_factor, tk_image, image_on_canvas
        new_width = int(image_width * zoom_factor)
        new_height = int(image_height * zoom_factor)
        x_offset= canvas.winfo_width()//2
        y_offset = canvas.winfo_height() // 2


        if event.delta > 0 and zoom_factor:  # Zoom in
            zoom_factor += zoom_step

        elif event.delta < 0 and zoom_factor > zoom_step:  # Zoom out
            zoom_factor -= zoom_step

        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        tk_image = ImageTk.PhotoImage(resized_image)

        canvas.delete(image_on_canvas)
        image_on_canvas = canvas.create_image(x_offset, y_offset,
                                              anchor="center", image=tk_image)

        canvas.config(scrollregion=canvas.bbox("all"))
    canvas.bind("<MouseWheel>", zoom)

    def on_button_press(event):
        nonlocal start_x, start_y
        start_x = event.x
        start_y = event.y

    def on_move_press(event):
        nonlocal start_x, start_y
        delta_x = event.x - start_x
        delta_y = event.y - start_y
        canvas.move(image_on_canvas, delta_x, delta_y)
        start_x = event.x
        start_y = event.y

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)