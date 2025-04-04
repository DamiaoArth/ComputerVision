import numpy as np
import tensorflow as tf
import cv2

# Load the saved model
best_model_file = "maskmodel/bombardilo/lungUnet.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

# Define image dimensions
Width = 256
Height = 256

# Load test image
testImagePath = "img.png"
img = cv2.imread(testImagePath)

# Resize the image
img2 = cv2.resize(img, (Width, Height))

# Normalize the image
img2 = img2 / 255.0

# Expand dimensions for model input
imgForModel = np.expand_dims(img2, axis=0)

# Predict mask
p = model.predict(imgForModel)
resultMask = p[0]
print(resultMask.shape)

# Convert mask to binary (black and white)
resultMask[resultMask <= 0.5] = 0
resultMask[resultMask > 0.5] = 255

# Resize and display settings
scale_percent = 60  # Scale percentage for resizing

w = int(img.shape[1] * scale_percent / 100)
h = int(img.shape[0] * scale_percent / 100)

dim = (w, h)

# Resize the image
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
mask = cv2.resize(resultMask, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Image", img)
cv2.imshow("Mask", mask) 
cv2.waitKey(0)
