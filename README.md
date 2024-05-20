Đây là dự án xây dựng một ứng dụng chỉnh sửa ảnh đơn giản ứng dụng deep learning.
Bao gồm các chức năng cơ bản như cắt, xoay, vẽ lên ảnh, có các filter, và có thể chuyển đổi màu sắc các bộ phận trên khuôn mặt, cùng với đó là thay đổi background. Ngoài ra chương trình cũng có chức năng thêm như chuyển thành ảnh ascii và photomosaic
Dự án đã áp dụng 2 model q là model phát hiện các bộ phận trên khuôn mặt(mắt, tóc, môi ... ) từ đó thay đổi màu sắc theo mong muốn model này train dựa trên kiến trúc Unet và được train trên tập dữ liệu Lapa dataset.
Model thứ 2 là model humansegmentation dùng để tách người ra khỏi nên, model này được train dựa trên kiến trúc Resnet50.
