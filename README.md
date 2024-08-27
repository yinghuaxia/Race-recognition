# Race Recognition using CSL-YOLO
Github : [CSL-YOLO](https://github.com/D0352276/CSL-YOLO) 

This project aims to use CSL-YOLO to complete race recognition tasks, and the detected faces will be classified into four races - White, Black, Asian, and Indian. Besides some main modifications in CSL-YOLO, there are also some tools to help the whole task go more smoothly. Let's start with some useful tools first.

### tools.py
This is an easy tool that helps labeling more easily. Images and their ground truth (the ground truth format corresponds to CSL-YOLO) should be placed in the same folder (for instance, folder A), with ground truth placed in the folder named json under folder A. After compilation, the image with the corresponding bounding box will show, and there are some ways to modify the labels.
1. The bounding box can be created by using the mouse to drag on the image, the class can be typed in the terminal. Press **N** to go to the next image.
2. Press **E** to enter the edit mode, **A**, and **D** to select the bounding box that must be edited. Press **E** again to confirm the bounding box to edit, and type the new class in the terminal, press **X** to delete the selected bounding box. Press enter to confirm the modification and press **Q** to exit the edit mode.
3. Press **N** for the next image and **B** for the previous image.

### format_check.ipynb
Functions in this file help check whether the created ground truth json files are valid or not. Details of functions are annotated above every function in this file. 

### random_paste_images.ipynb
This file pastes images (from 1 to 4) to a background image, and created corresponding new ground truth of the new image. The resized shape of the pasted image has considered the proportion of anchors and the bounding box of the image. It can also add augmentations to every pasted image, the pasted image is assumed of square shape to prevent over-deformation.

![One person pasted](https://github.com/user-attachments/assets/c9eb7a65-306a-4582-a434-9d8551189124)
![Two people pasted](https://github.com/user-attachments/assets/618fd258-58e4-42d7-9bda-ca86369ef256)

![Three people pasted](https://github.com/user-attachments/assets/3641b43f-0ca2-4471-b198-2d696be6b6e0)
![Four people pasted](https://github.com/user-attachments/assets/f2ff934a-68c8-454e-90d5-0208fec699b3)
