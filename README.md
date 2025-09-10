# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:

### Step 1  
Read the input image using OpenCV and convert it from BGR to RGB format.  

### Step 2  
Apply an averaging filter using a normalized box filter (simple smoothing).  

### Step 3  
Apply a weighted averaging filter to preserve edges better while reducing noise.  

### Step 4  
Apply Gaussian and Median filters for advanced smoothing operations.  

### Step 5  
Apply a sharpening filter (like Laplacian or custom kernel) to enhance the edges in the image.  


## Program:
### Developed By   :KARUNIYA M
### Register Number:212223240068
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("q1.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel = np.ones((5,5), np.float32) / 25
avg_img = cv2.filter2D(image_rgb, -1, kernel)

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(avg_img)
plt.title("Averaging Filter")
plt.axis("off")
plt.show()


```
ii) Using Weighted Averaging Filter
```
kernel_w = np.array([[1,2,1],
                     [2,4,2],
                     [1,2,1]]) / 16
weighted_img = cv2.filter2D(image_rgb, -1, kernel_w)

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(weighted_img)
plt.title("Weighted Average Filter")
plt.axis("off")
plt.show()

```
iii) Using Gaussian Filter
```
gaussian_img = cv2.GaussianBlur(image_rgb, (5,5), 0)
plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_img)
plt.title("Gaussian Filter")
plt.axis("off")
plt.show()

```
iv)Using Median Filter
```
median_img = cv2.medianBlur(image_rgb, 5)

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(median_img)
plt.title("Median Filter")
plt.axis("off")
plt.show()



```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("q1.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel_sharpen = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
sharpen_img = cv2.filter2D(image_rgb, -1, kernel_sharpen)

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharpen_img)
plt.title("Laplacian Linear Kernel")
plt.axis("off")
plt.show()




```
ii) Using Laplacian Operator
```
laplacian_img = cv2.Laplacian(image_rgb, cv2.CV_64F)
laplacian_img = np.uint8(np.absolute(laplacian_img))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(laplacian_img, cmap='gray')
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()





```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter

<img width="380" height="455" alt="image" src="https://github.com/user-attachments/assets/8f86d778-2704-470b-b0db-1197f7f9d0e6" />


ii)Using Weighted Averaging Filter

<img width="357" height="472" alt="image" src="https://github.com/user-attachments/assets/14af98ca-8a67-4580-9b86-9c931d55d1e2" />


iii)Using Gaussian Filter

<img width="380" height="468" alt="image" src="https://github.com/user-attachments/assets/14fbfa37-2110-4680-9316-a9cd184864c9" />


iv) Using Median Filter

<img width="364" height="470" alt="image" src="https://github.com/user-attachments/assets/8cd0a167-bd24-401f-bcd0-aadcd3653350" />


### 2. Sharpening Filters
</br>

<img width="823" height="773" alt="image" src="https://github.com/user-attachments/assets/c84bda98-db1b-41af-b6dc-a74baf35efb7" />


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
