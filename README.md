# Race Recognition using CSL-YOLO
Github : [CSL-YOLO](https://github.com/D0352276/CSL-YOLO) 

This project aims to use CSL-YOLO to complete race recognition tasks, detected faces will be classified into four races - White, Black, Asian, and Indian. Besides some main modification in CSL-YOLO, 
there are also some tools to help the whole task to go more smoothly. Let's start by some useful tools first.

### tools.py
This is an easy tool that helps labeling more easily. Images and its ground truth (the format of ground truth is correspond to CSL-YOLO) should be placed in the same folder (for instance, folder A), with ground truth placed in the folder named json under folder A. After compile, the image with corresponding bounding box will show, and there are some ways to modify the labels.
1. The bounding box can be created by simply use the mouse to drag on the image, the class can be typed in the terminal. Press **N** to go to the next image.
2. Press **E** to enter the edit mode, **A** and **D** to select the bounding box that needs to be edit. Press **E** again to confirm the bounding box to edit, and type the new class in the terminal, press enter to comfirm the modification and press **Q** to exit the edit mode.
3. Press **N** for the next image and **B** for the previous image
