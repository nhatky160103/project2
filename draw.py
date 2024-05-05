import io
import tkinter as tk
from tkinter import filedialog, colorchooser
import customtkinter
import cv2
from PIL import Image
import numpy as np
pen_color = "red"
pen_size = 2
canvas_image = None


def start_draw(event):
    draw.is_drawing = True
    draw.last_x = event.x
    draw.last_y = event.y


def end_draw(event):
    draw.is_drawing = False


def draw(event, canvas):

    if draw.is_drawing:
        if hasattr(draw, 'last_x') and hasattr(draw, 'last_y'):
            x1, y1 = draw.last_x, draw.last_y
            x2, y2 = event.x, event.y
            canvas.create_line(x1, y1, x2, y2, fill=pen_color, width=pen_size, tags="drawing")
            draw.last_x = event.x
            draw.last_y = event.y



def change_color():
    global pen_color
    pen_color = colorchooser.askcolor(title="Select pen color")[1]


def on_scrollbar_change(value):
    global pen_size
    pen_size = value


def clear_canvas(canvas):
    canvas.delete("drawing")



def create_draw_frame(canvas, frame):
    for widget in frame.winfo_children():
        widget.destroy()
    global canvas_image  # Đảm bảo biến canvas_image là global

    print("create a frame")

    color_button = customtkinter.CTkButton(frame, text="Change pen color", command=change_color)
    color_button.pack(pady=10)

    def update_label(event):
        value_label.config(text=str(int(pen_size_slider.get())))

    pen_size_slider = customtkinter.CTkSlider(frame, from_=0, to=15, command=on_scrollbar_change)
    pen_size_slider.pack(pady=10)
    pen_size_slider.set(0)
    pen_size_slider.bind("<B1-Motion>",update_label )
    value_label = tk.Label(frame, text=str(pen_size_slider.get()),  font=("Arial", 16))
    value_label.pack(pady=5)




    clear_button = customtkinter.CTkButton(frame, text="Erase", command=lambda: clear_canvas(canvas))
    clear_button.pack(pady=10)
    # Tạo một biến cờ để theo dõi việc vẽ
    draw_finished = False

    # Hàm gọi khi kết thúc vẽ
    def finish_draw(event):
        nonlocal draw_finished
        draw_finished = True

    # canvas.bind("<Button-1>", start_draw)
    # canvas.bind("<B1-Motion>", lambda event: draw(event, canvas))
    # canvas.bind("<ButtonRelease-1>", end_draw)
    canvas.bind("<Button-1>", start_draw)
    canvas.bind("<B1-Motion>", lambda event: draw(event, canvas))
    canvas.bind("<ButtonRelease-1>", lambda event: [end_draw(event), finish_draw(event)])


    while not draw_finished:
        canvas.update()

    ps_data = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(ps_data.encode('utf-8')))
    img_rgb = img.convert('RGB')

    img_rgb_array = np.array(img_rgb)
    result = cv2.cvtColor(img_rgb_array, cv2.COLOR_BGR2RGB)
    return result





