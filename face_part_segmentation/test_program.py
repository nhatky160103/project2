import cv2
import numpy as np

def undistort_image(image_path, camera_matrix, distortion_coeffs, scale=1.0):
    img = cv2.imread(image_path)

    h, w = img.shape[:2]

    new_camera_matrix = scale * camera_matrix

    undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs, None, new_camera_matrix)

    return undistorted_img


camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]])
distortion_coeffs = np.array([0.1, 0.1, 0, 0, 0], dtype=np.float32)


image_path = "test_image/test.jpg"

scale_factor = 0.3


for i in range(7):
    scale_factor+= 0.1
    undistorted_image = undistort_image(image_path, camera_matrix, distortion_coeffs, scale=scale_factor)

    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
