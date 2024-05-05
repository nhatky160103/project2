import tkinter as tk
import customtkinter
import tkinter as tk
from PIL import Image, ImageOps, ImageTk, ImageFilter
from PIL import Image, ImageTk
import numpy as np
import cv2
from draw import create_draw_frame
from filter import apply_filter, update_value, edge_detection, change_to_threshold, change_blur, change_contrast
from tkinter import messagebox, PhotoImage, Button, ttk, filedialog, Label, colorchooser
from ascii import create_ascii_image
from photomosaic import create_photomosaic
from cut_rotate import create_rotate, ImageCutter, zoom_image
from change_color_2 import change_color_face
import tensorflow as tf





customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light")
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def change_appearance_mode(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)

def change_scaling(new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    customtkinter.set_widget_scaling(new_scaling_float)


root = customtkinter.CTk()
root.title("PHOTO EDITOR APP")
root.geometry(f"{1100}x{580}")

pen_color = "red"
pen_size = 2
file_path = ""
saved_image= None

origin_image= None
canvas_image = None
color_image=None
init_image=None

def save_canvas_as_image():
    global saved_image

    if saved_image is not None:
        saved_image = saved_image.astype(np.uint8)
        new_file_name = filedialog.asksaveasfilename(defaultextension=".jpg")
        cv2.imwrite(new_file_name, saved_image)
        messagebox.showinfo("success", "The image saved ")
    else:
        if file_path!= "":
            saved_image= cv2.imread(file_path)
            new_file_name = filedialog.asksaveasfilename(defaultextension=".jpg")
            cv2.imwrite(new_file_name, saved_image)
            messagebox.showinfo("success", "The image saved ")
        else:
            messagebox.showinfo("failed", "You don't have an image")



def add_image():

    global file_path, canvas_image, origin_image, color_image, saved_image,  init_image
    temp= file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/java_workspace/face_segmentation/lapa/LaPa/test/images")
    if not file_path:
        file_path= temp

    if file_path!="":
        init_image =saved_image = color_image= cv2.imread(file_path)

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


        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        canvas_image = ImageTk.PhotoImage(resized_image)


        canvas.delete("all")
        canvas.create_image(x_offset, y_offset, image=canvas_image, anchor="nw")

        or_image = Image.open(file_path)
        or_image.thumbnail((200, 200))
        origin_image = ImageTk.PhotoImage(or_image)
        original_image_label = customtkinter.CTkLabel(original_image_frame, text="Original Image:", anchor="center")
        original_image_label.grid(row=0, column=0,sticky="nsew")
        Label(original_image_frame, image=origin_image).grid(row=1, column=0, sticky="nsew")
        original_image_frame.columnconfigure(0, weight=1)
        original_image_frame.rowconfigure((0,1), weight=1)



def update_image_canvas(event):
    global file_path, canvas_image, origin_image, color_image, saved_image

    if file_path != "":
        image= cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
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

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        canvas_image = ImageTk.PhotoImage(resized_image)

        canvas.delete("all")
        canvas.create_image(x_offset, y_offset, image=canvas_image, anchor="nw")


model = tf.keras.models.load_model("models/model_29_3.h5")
label_dict = {
    "background": 0,
    "skin": 1,
    "eyebrow": 2,
    "eye": 4,
    "nose": 6,
    "lip": 7,
    "inner mouth": 8,
    "hair": 10,
}
def find_mask(image, num_label):
    global  model
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # [[0,1,0][1,1,1][0,1,0]]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def convert_to_mask(y, num_label):
    if num_label == 0 or num_label == 1 or num_label == 6 or num_label == 8 or num_label == 10:
        for i in range(512):
            for j in range(512):
                if y[i][j] == num_label:
                    y[i][j] = 255
                else:
                    y[i][j] = 0
        return y.astype(np.uint8)
    elif num_label == 2:
        for i in range(512):
            for j in range(512):
                if y[i][j] == 2 or y[i][j] == 3:
                    y[i][j] = 255
                else:
                    y[i][j] = 0
        return y.astype(np.uint8)
    elif num_label == 4:
        for i in range(512):
            for j in range(512):
                if y[i][j] == 4 or y[i][j] == 5:
                    y[i][j] = 255
                else:
                    y[i][j] = 0
        return y.astype(np.uint8)
    elif num_label == 7:
        for i in range(512):
            for j in range(512):
                if y[i][j] == 7 or y[i][j] == 9:
                    y[i][j] = 255
                else:
                    y[i][j] = 0
        return y.astype(np.uint8)





def on_scale_changed(value):
    global saved_image
    value = int(value)
    saved_image=update_value(value, file_path, canvas)
def on_scale_changed2(value):
    global saved_image
    value = int(value)
    saved_image=edge_detection(file_path, canvas,  value)

def on_scale_changed3(value):
    global saved_image
    value = int(value)
    saved_image= change_blur(file_path, canvas,  value)
def on_scale_changed4(value):
    global  saved_image
    value = int(value)
    saved_image= create_rotate(file_path,canvas, value)

def on_scale_changed5(value):
    global saved_image
    value = int(value)
    saved_image= change_contrast(file_path, canvas, value)
def on_scale_changed6(value):
    global  saved_image
    saved_image = create_photomosaic(file_path, canvas, value)
def on_scale_change7():
    global saved_image
    saved_image= create_draw_frame(canvas, custom_frame)

def on_scale_change8(value):
    global saved_image
    saved_image = apply_filter(value,file_path, canvas)


def on_scale_change9(value):
    global saved_image
    value= int(value)
    saved_image = create_ascii_image(file_path, canvas, value, 1)
def on_scale_change10(value):
    global saved_image
    value= int(value)
    saved_image = create_ascii_image(file_path, canvas, value, 0)

def filter():
    for widget in custom_frame.winfo_children():
        widget.destroy()
    if file_path =="":
        messagebox.showinfo("warning", "You have not imported any images yet!")
        add_image()

    filter_label = customtkinter.CTkLabel(custom_frame, text="Select Filter")
    filter_label.pack(pady= 10)

    filter_combobox =  customtkinter.CTkComboBox(custom_frame, values=["Black and White", "Blur",
                                                    "Emboss", "Sharpen", "Smooth","sobelx", "sobely","identity","edge_detection","sharpen", "vignette"]
                                                 ,command= on_scale_change8
                                                 )
    filter_combobox.pack(pady=5)


    slider_label = customtkinter.CTkLabel(custom_frame, text="Change light")
    slider_label.pack(pady=7)

    slider = customtkinter.CTkSlider(custom_frame, from_=0, to=25, command=on_scale_changed)
    slider.pack(pady=5, padx=(5,5))


    slider_label = customtkinter.CTkLabel(custom_frame, text="edge change")
    slider_label.pack(pady=7)
    slider = customtkinter.CTkSlider(custom_frame, from_=1, to=20, command=on_scale_changed2)
    slider.pack(pady=5, padx=(5,5))

    slider_label =customtkinter.CTkLabel(custom_frame, text="change blur")
    slider_label.pack(pady=7)
    slider = customtkinter.CTkSlider(custom_frame, from_=1, to=40, command=on_scale_changed3)
    slider.pack(pady=5, padx=(5,5))

    slider_label = customtkinter.CTkLabel(custom_frame, text="change contrast")
    slider_label.pack(pady=7)
    slider = customtkinter.CTkSlider(custom_frame, from_=1, to=20, command=on_scale_changed5)
    slider.pack(pady=5, padx=(5,5))

    button =customtkinter.CTkButton(custom_frame, text='Binary image', command=lambda :change_to_threshold(file_path, canvas))
    button.pack(pady=10)


def change_to_ascii():
    for widget in custom_frame.winfo_children():
        widget.destroy()
    if file_path =="":
        messagebox.showinfo("warning", "You have not imported any images yet!")
        add_image()

    slider_label = customtkinter.CTkLabel(custom_frame, text="Change char size")
    slider_label.pack(pady=5)

    def update_label(value):
        value_label.config(text=str(int(value)))


    slider = customtkinter.CTkSlider(custom_frame, from_=2, to=7, command=update_label)
    slider.pack(pady=10)
    slider.set(2)
    value_label = tk.Label(custom_frame, text=str(slider.get()),  font=("Arial", 16))
    value_label.pack()

    button = customtkinter.CTkButton(custom_frame, text="Convert with color!", command=lambda:on_scale_change9(slider.get()))
    button.pack(pady=15)

    button = customtkinter.CTkButton(custom_frame, text="Convert no color!", command=lambda: on_scale_change10(slider.get()))
    button.pack(pady=15)

def change_to_photomosaic():
    for widget in custom_frame.winfo_children():
        widget.destroy()
    if file_path == "":
        messagebox.showinfo("warning", "You have not imported any images yet!")
        add_image()

    def update_label(value):
        value_label.config(text=str(int(value)))




    slider_label = customtkinter.CTkLabel(custom_frame, text="Change sub image size")
    slider_label.pack(pady=5)
    slider =  customtkinter.CTkSlider(custom_frame, from_=5, to=20 , command=update_label)
    slider.pack(pady=10)
    slider.set(5)
    value_label = tk.Label(custom_frame, text=str(slider.get()), font=("Arial", 16))
    value_label.pack()


    button = customtkinter.CTkButton(custom_frame, text="Convert!", command=lambda:on_scale_changed6(int(slider.get())))
    button.pack(pady=15)

def rotate():
    for widget in custom_frame.winfo_children():
        widget.destroy()
    if file_path == "":
        messagebox.showinfo("warning", "You have not imported any images yet!")
        add_image()

    slider_label = customtkinter.CTkLabel(custom_frame, text="Change angle")
    slider_label.pack(pady=5)

    def update_label(event):
        value_label.config(text=str(int(slider.get())))

    slider = customtkinter.CTkSlider(custom_frame, from_=0, to=360, command=on_scale_changed4)
    slider.pack(pady=10, padx=(5,5))
    slider.set(0)
    slider.bind("<B1-Motion>",update_label)
    value_label = tk.Label(custom_frame, text=str(slider.get()), font=("Arial", 16))
    value_label.pack()




is_on = False
on = PhotoImage(file="on.png")
off = PhotoImage(file="off.png")

# Khai báo biến on_button ở global scope
on_button = None
image_cutter= None
def switch():
    global is_on, on_button

    # Determine is on or off
    if is_on:
        on_button.config(image=off)
        is_on = False
        canvas.delete("rect")
        canvas.unbind("<ButtonPress-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<ButtonRelease-1>")
        canvas.delete("rect")
        canvas.config(cursor="")
    else:
        on_button.config(image=on)
        is_on = True
        ImageCutter(file_path, canvas)




def cut():
    global on_button  # Khai báo biến on_button ở global scope
    for widget in custom_frame.winfo_children():
        widget.destroy()
    if file_path == "":
        messagebox.showinfo("warning", "You have not imported any images yet!")
        add_image()


    cut_label = customtkinter.CTkLabel(custom_frame, text="Cut mode")
    cut_label.pack(pady=10)

    on_button = Button(custom_frame, image=off, bd=0, command=switch)
    on_button.pack(pady=10)


listbox = None
color_combobox = None

def show_color_box():
    global listbox, color_combobox
    item= ""
    if listbox.get() :
        item = listbox.get()

    if item:
        choose_color = colorchooser.askcolor(title="Select color")[0]
        on_scale_change_color(item,choose_color)


def on_scale_change_color(item, value):

    global  saved_image, color_image,  init_image
    mask = find_mask(init_image ,label_dict[item])
    saved_image = color_image = change_color_face(canvas, color_image, item, value, mask)
def clear_image_color():
    global saved_image, color_image
    saved_image = color_image= org_img= cv2.imread(file_path)
    mask = find_mask(org_img,0)
    saved_image = color_image = change_color_face(canvas, org_img, "background", (0,0,0),mask )


change_color_button= None
def show_button(event):
    change_color_button.pack(pady=5)


def change_color():
    global  listbox, change_color_button
    for widget in custom_frame.winfo_children():
        widget.destroy()
    if file_path == "":
        messagebox.showinfo("warning", "You have not imported any images yet!")
        add_image()

    listbox = customtkinter.CTkComboBox(custom_frame, values=["hair", "background", "skin", "eyebrow", "eye",  "nose", "lip", "mouth"], command=show_button)
    listbox.pack(side=tk.TOP, pady=5)
    # # Thêm mục vào danh sách
    # items = ["hair", "background", "skin", "eyebrow", "eye",  "nose", "lip", "mouth"]
    # for item in items:
    #     listbox.insert(tk.END, item)
    # listbox.config(height=len(items))

    change_color_button = customtkinter.CTkButton(custom_frame, text="Change color", command= show_color_box)
    listbox.bind("<Button-1>", show_button)
    clear_color_button = customtkinter.CTkButton(custom_frame, text="clear all", command=clear_image_color)
    clear_color_button.pack(pady=5)




root.grid_columnconfigure((1, 2, 3), weight=1)
root.grid_rowconfigure((0, 1, 2), weight=1)

left_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
left_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
left_frame.grid_rowconfigure(4, weight=1)

right_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
right_frame.grid(row=1, column=4, rowspan=3, sticky="nsew", padx=(0,10), pady=(0,10))
right_frame.grid_rowconfigure(4, weight=1)


original_image_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
original_image_frame.grid(row=0, column=4, sticky="nsew", padx=(0,10), pady=(0,10))




logo_label = customtkinter.CTkLabel(left_frame, text="TOOLBAR", font=customtkinter.CTkFont(size=20, weight="bold"))
logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

button_open = customtkinter.CTkButton(left_frame, text="Mở ảnh", width=110, height=30, command= add_image)
button_save = customtkinter.CTkButton(left_frame, text="Lưu ảnh", width=110, height=30, command=save_canvas_as_image)
button_open.grid(row=1, column=0, padx=10, pady=10)
button_save.grid(row=2, column=0, padx=10, pady=10)

custom_frame = customtkinter.CTkFrame(left_frame, width=150, corner_radius=0)
custom_frame.grid(row=3, rowspan=10, sticky="nsew", pady=10, padx=5)

appearance_mode_label = customtkinter.CTkLabel(right_frame, text="Appearance Mode:", anchor="w")
appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))

appearance_mode_optionmenu = customtkinter.CTkOptionMenu(right_frame, values=["Light", "Dark", "System"], command=change_appearance_mode)
appearance_mode_optionmenu.grid(row=7, column=0, padx=20, pady=(10, 10))

scaling_label = customtkinter.CTkLabel(right_frame, text="UI Scaling:", anchor="w")
scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))

scaling_optionmenu = customtkinter.CTkOptionMenu(right_frame, values=["80%", "90%", "100%", "110%", "120%"], command=change_scaling)
scaling_optionmenu.grid(row=9, column=0, padx=20, pady=(10, 20))



canvas = customtkinter.CTkCanvas(root, background="gray", borderwidth=2, relief="solid")
canvas.grid(row=0, column=1, columnspan=3, rowspan=3, sticky="nsew", padx=30, pady=30)
canvas.bind("<Configure>",update_image_canvas)



menubar = tk.Menu(root)
file = tk.Menu(menubar, tearoff=0)
file.add_command(label='New image')
file.add_command(label='Save')
file.add_separator()
file.add_command(label='Exit', command = root.destroy)
menubar.add_cascade(label='File', menu=file)

edit = tk.Menu(menubar, tearoff=0)
edit.add_command(label='Draw', command=on_scale_change7)
menubar.add_cascade(label='Draw', menu=edit)

view_menu = tk.Menu(menubar, tearoff=0)
view_menu.add_command(label="Apply filter",command=filter)
view_menu.add_command(label="Rotate",  command=rotate)
view_menu.add_command(label="Cut", command=cut)
menubar.add_cascade(label="Custom", menu=view_menu)

tools_menu = tk.Menu(menubar, tearoff=0)
tools_menu.add_command(label="Ascii", command=change_to_ascii)
tools_menu.add_command(label="Photomosaic", command=change_to_photomosaic)
menubar.add_cascade(label="Tools", menu=tools_menu)

tools_menu = tk.Menu(menubar, tearoff=0)
tools_menu.add_command(label="Change color", command=change_color)
menubar.add_cascade(label="Color", menu=tools_menu)

help_ = tk.Menu(menubar, tearoff=0)
help_.add_command(label='Tk Help', command=None)
help_.add_command(label='Demo', command=None)
help_.add_separator()
help_.add_command(label='About Tk', command=None)
menubar.add_cascade(label='Help', menu=help_)

root.config(menu=menubar)

root.mainloop()
