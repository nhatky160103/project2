Đây là dự án xây dựng một ứng dụng chỉnh sửa ảnh đơn giản ứng dụng deep learning.

Bao gồm các chức năng cơ bản như cắt, xoay, vẽ lên ảnh, có các filter, và có thể chuyển đổi màu sắc các bộ phận trên khuôn mặt, cùng với đó là thay đổi background. Ngoài ra chương trình cũng có chức năng thêm như chuyển thành ảnh ascii và photomosaic

Dự án đã áp dụng 2 model một là model phát hiện các bộ phận trên khuôn mặt(mắt, tóc, môi ... ) từ đó thay đổi màu sắc theo mong muốn model này train dựa trên kiến trúc Unet và được train trên tập dữ liệu Lapa dataset.

Kết quả:
![img.png](img.png)
![img_1.png](img_1.png)

Model thứ hai là model humansegmentation dùng để tách người ra khỏi nền, model này được train dựa trên kiến trúc Resnet50.
kết quả của hai model có thể xem ở hình dưới

Kết quả:
![result_total2.png](..%2F..%2F..%2FSubject%2Flatex%2FIMAGE%2FChapter2%2Fresult_total2.png)
![result_total3.png](..%2F..%2F..%2FSubject%2Flatex%2FIMAGE%2FChapter2%2Fresult_total3.png)
![result_total.png](..%2F..%2F..%2FSubject%2Flatex%2FIMAGE%2FChapter2%2Fresult_total.png)

Loss và accuracy:
![train_val_loss.png](..%2F..%2F..%2FSubject%2Flatex%2FIMAGE%2FChapter2%2Ftrain_val_loss.png)
![train_val_accuracy.png](..%2F..%2F..%2FSubject%2Flatex%2FIMAGE%2FChapter2%2Ftrain_val_accuracy.png)

Thay đổi màu tóc<br>
<img src="https://github.com/nhatky160103/project2/assets/118037343/fb9869d8-cd86-4dc0-95dc-237339e33ad5" width="500" height="500" />

Thay đổi màu môi<br>
<img src="https://github.com/nhatky160103/project2/assets/118037343/b1f54057-c740-4ad1-825d-c6d7d9605d61" width="500" height="500" />

 Thay đổi màu lông mày<br>
<img src="https://github.com/nhatky160103/project2/assets/118037343/143f0710-9a71-47b6-b22b-2465bfebea4e" width="500" height="500" />

Thay đổi màu mắt<br>
<img src="https://github.com/nhatky160103/project2/assets/118037343/4cda80fc-4b0d-4185-840c-df577718c6cf" width="500" height="500" />

 Thay đổi nền<br>
 <img src="https://github.com/nhatky160103/project2/assets/118037343/41232d0f-a5e1-46e9-9365-b1e485452420"  width="500"  />

