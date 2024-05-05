import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

from change_color import change_color_face

def main():
    root = tk.Tk()
    root.title("Change Color")

    # Hàm sẽ gọi khi các thanh trượt được điều chỉnh
    def update_color():
        r= int(scale_r.get())
        g=int(scale_g.get())
        b=int(scale_b.get())
        result= change_color_face(10, (r,g,b))

        cv2.imshow("image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Tạo thanh trượt
    scale_r = tk.Scale(root, from_=1, to=255, orient=tk.HORIZONTAL, label="Color Level",
                     length=300)
    scale_r.set(128)
    scale_r.pack()
    scale_g = tk.Scale(root, from_=1, to=255, orient=tk.HORIZONTAL, label="Color Level",
                     length=300)
    scale_g.set(128)
    scale_g.pack()
    scale_b = tk.Scale(root, from_=1, to=255, orient=tk.HORIZONTAL, label="Color Level",
                     length=300)
    scale_b.set(128)
    scale_b.pack()
    button  = tk.Button(root, bg="white", command=update_color)
    button.pack()


    root.mainloop()

if __name__ == "__main__":
    main()
