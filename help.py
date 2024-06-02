import tkinter as tk
from PIL import Image, ImageTk
def show_help(root):
    help_window = tk.Toplevel(root)
    help_window.title("Tk Help")
    instruction_text = """Hướng dẫn sử dụng:
    1. Mở ảnh: Sử dụng chức năng "Mở ảnh" để tải ảnh cần chỉnh sửa.
    2. Chọn chức năng chỉnh sửa: Sử dụng các nút chức năng để cắt, xoay, vẽ, áp dụng bộ lọc, thay đổi màu sắc, và thay đổi nền.
    3. Lưu ảnh: Sử dụng chức năng "Lưu ảnh" để lưu lại kết quả chỉnh sửa.
    4. Thử nghiệm thêm: Sử dụng các chức năng chuyển đổi thành ảnh ASCII và photomosaic để thử nghiệm các hiệu ứng đặc biệt."""

    tk.Label(help_window, text=instruction_text, font=("Arial", 18), justify="left", wraplength=600).pack(padx=10,
                                                                                                          pady=10,
                                                                                                          fill='x')


def show_demo(root):
    demo_window = tk.Toplevel(root)
    demo_window.title("Demo")

    # Text description
    description = """Đây là dự án xây dựng một ứng dụng chỉnh sửa ảnh đơn giản ứng dụng deep learning.

Bao gồm các chức năng cơ bản như:
- Cắt: Cho phép bạn cắt một phần của ảnh.
- Xoay: Xoay ảnh theo các góc độ khác nhau.
- Vẽ lên ảnh: Vẽ tự do lên ảnh.
- Bộ lọc (Filter): Áp dụng các bộ lọc để thay đổi màu sắc và cảm giác của ảnh.
- Chuyển đổi màu sắc các bộ phận trên khuôn mặt: Thay đổi màu tóc, mắt, môi, v.v.
- Thay đổi nền (Background): Tách người ra khỏi nền và thay đổi nền ảnh.

Ứng dụng còn có các chức năng thêm như:
- Chuyển đổi thành ảnh ASCII.
- Photomosaic: Tạo ảnh mosaic từ nhiều ảnh nhỏ.

Dự án sử dụng hai model:
1. Model phát hiện các bộ phận trên khuôn mặt (mắt, tóc, môi, v.v.) để thay đổi màu sắc theo mong muốn. Model này dựa trên kiến trúc Unet và được train trên tập dữ liệu Lapa dataset.
2. Model human segmentation dùng để tách người ra khỏi nền, dựa trên kiến trúc Resnet50.

Dưới đây là một số kết quả của hai model:"""

    tk.Label(demo_window, text=description, font=("Arial", 16), justify="left", wraplength=600).pack(padx=10, pady=10,
                                                                                                     fill='x')

    # Load images
    try:
        img1 = Image.open("results/change_background.jpg")
        width, height = img1.size
        img1 = img1.resize((300, int(300 * height / width)), Image.LANCZOS)
        img1 = ImageTk.PhotoImage(img1)

        img2 = Image.open("results/hair.png")
        width, height = img2.size
        img2 = img2.resize((300, int(300 * height / width)), Image.LANCZOS)
        img2 = ImageTk.PhotoImage(img2)

        img3 = Image.open("results/lip.png")
        width, height = img3.size
        img3 = img3.resize((300, int(300 * height / width)), Image.LANCZOS)
        img3 = ImageTk.PhotoImage(img3)

        # Display images
        img_frame = tk.Frame(demo_window)
        img_frame.pack(pady=10)

        img_label1 = tk.Label(img_frame, image=img1)
        img_label1.image = img1  # keep a reference!
        img_label1.pack(side="left", padx=10)

        img_label2 = tk.Label(img_frame, image=img2)
        img_label2.image = img2  # keep a reference!
        img_label2.pack(side="left", padx=10)

        img_label3 = tk.Label(img_frame, image=img3)
        img_label3.image = img3  # keep a reference!
        img_label3.pack(side="left", padx=10)



    except Exception as e:
        tk.Label(demo_window, text="Error loading images: " + str(e), font=("Arial", 12), fg="red").pack(pady=10)


def show_about(root):
    about_window = tk.Toplevel(root)
    about_window.title("About Us")
    tk.Label(about_window, text="""Đề tài : xây dựng ứng dụng chỉnh sửa ảnh ứng dụng deep learning. 
    Họ và tên học sinh: Đinh Nhật Ký
    Giáo viên hướng dẫn: Lê Thị Hoa
    Trường công nghệ thông tin và truyền thông.
    """, font=25).pack(padx=200, pady=200)
    tk.Label(about_window, text="Xem thêm tài liệu hoặc mã nguồn tại:", font=("Arial", 16, "bold")).pack(pady=10)
    tk.Label(about_window, text="GitHub Repository: https://github.com/nhatky160103/project2", font=("Arial", 16), fg="blue",
             cursor="hand2").pack()
    tk.Label(about_window, text="Liên hệ: ky.dn215410@sis.hust.edu.vn", font=("Arial", 16)).pack(pady=10)