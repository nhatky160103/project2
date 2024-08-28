Project: Simple Image Editing Application Using Deep Learning
This project involves creating a user-friendly image editing application utilizing deep learning techniques to provide advanced image manipulation capabilities.

Core Features:
Basic Functionalities: Includes cropping, rotating, drawing on images, applying various filters, and changing the colors of specific facial features.
Advanced Features:
Facial Feature Color Adjustment: Modify the colors of hair, lips, eyebrows, and eyes.
Background Replacement: Change the background of an image.
Additional Image Transformations: Convert images to ASCII art and photomosaic.
Models Used:
Facial Feature Detection Model:

Purpose: Detects facial features such as eyes, hair, lips, and allows for color modifications based on user preferences.
Architecture: Built using the U-Net architecture, a popular choice for image segmentation tasks.
Training Data: Trained on the LAPA dataset, which includes diverse facial images.
Performance:
Results: Demonstrates the model’s effectiveness in identifying and modifying facial features.
![result_total.png](results%2Fresult_total.png)
![result_total2.png](results%2Fresult_total2.png)

Loss và accuracy:

![train_val_accuracy.png](results%2Ftrain_val_accuracy.png)
![train_val_loss.png](results%2Ftrain_val_loss.png)



Human Segmentation Model:

Purpose: Segments the human figure from the background, enabling background changes and further image manipulations.
Architecture: Utilizes ResNet50, known for its deep residual learning framework that improves performance on complex tasks.
Performance:
Results: Shows the effectiveness of separating the human figure from the background.
Results:
![img.png](img.png)
![img_1.png](img_1.png)

Examples of Feature Changes:
Hair Color Change: <img src="https://github.com/nhatky160103/project2/assets/118037343/fb9869d8-cd86-4dc0-95dc-237339e33ad5" width="500" height="500" />
Lip Color Change: <img src="https://github.com/nhatky160103/project2/assets/118037343/b1f54057-c740-4ad1-825d-c6d7d9605d61" width="500" height="500" />
Eyebrow Color Change: <img src="https://github.com/nhatky160103/project2/assets/118037343/143f0710-9a71-47b6-b22b-2465bfebea4e" width="500" height="500" />
Eye Color Change: <img src="https://github.com/nhatky160103/project2/assets/118037343/4cda80fc-4b0d-4185-840c-df577718c6cf" width="500" height="500" />
Background Replacement: <img src="https://github.com/nhatky160103/project2/assets/118037343/41232d0f-a5e1-46e9-9365-b1e485452420" width="500" />


Developed an application that integrates advanced deep learning models to offer sophisticated image editing features.
Successfully trained models on complex datasets, achieving high accuracy in feature detection and segmentation.
This project demonstrates proficiency in applying deep learning techniques to practical image editing tasks and showcases the ability to implement and fine-tune models for specific use cases.





