
from draw import create_draw_frame, get_drawing
import cv2
import tkinter as tk
def main():
    def test():

        result = get_drawing(canvas)
        cv2.imshow("image", result)
        cv2.waitKey(0)

    root = tk.Tk()

    frame1 = tk.Frame(root, padx=20, pady=20, background="red")
    frame1.pack(pady=10)
    frame2 = tk.Frame(root, padx=20, pady=20, background="yellow")
    frame2.pack(pady=10)
    canvas = tk.Canvas(root, width=200, height=200)
    canvas.pack(pady=10)
    create_draw_frame(canvas, frame1)
    # Gắn kết sự kiện chuột trái với hàm left_click_function
    button = tk.Button(frame2, text="adafads", command=test)
    button.pack(pady=10)


    root.mainloop()


if __name__ == "__main__":
    main()
