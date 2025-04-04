import cv2
import numpy as np
import glob
from tqdm import tqdm

Height = 256
Width = 256

path = "/home/est.arthurmendes/Música/tralalerotralala/maskmodel/bombardilo"

imagesPath = path + "/CXR_png/*.png"  # Corrigido
leftMaskPath = path + "/ManualMask/leftMask/*.png"  # Corrigido
rightMaskPath = path + "/ManualMask/rightMask/*.png"  # Corrigido


print("Images in folder, left mask images, right mask images:")
listOfImages = glob.glob(imagesPath)
listOfLeftMaskImages = glob.glob(leftMaskPath)
listOfRightMaskImages = glob.glob(rightMaskPath)

print(len(listOfImages), len(listOfLeftMaskImages), len(listOfRightMaskImages))

img = cv2.imread(listOfImages[0], cv2.IMREAD_COLOR)
print(img.shape)

img = cv2.resize(img, (Height, Width))

left_mask = cv2.imread(listOfLeftMaskImages[0], cv2.IMREAD_GRAYSCALE)
right_mask = cv2.imread(listOfRightMaskImages[0], cv2.IMREAD_GRAYSCALE)

left_mask = cv2.resize(left_mask, (Height, Width))
right_mask = cv2.resize(right_mask, (Height, Width))

finalMask = left_mask + right_mask

cv2.imshow("Image", img)
cv2.imshow("Left Mask", left_mask)
cv2.imshow("Right Mask", right_mask)
cv2.imshow("Final Mask", finalMask)

#look at one mask
# reduce the size to see the values

mask16 = cv2.resize(finalMask, (16, 16))
print(mask16)

mask16[mask16 > 0] = 1
print("==================================")
print(mask16)

allImages=[]
maskImages=[]

print("start loading the train images and mask")
for imgFile, leftMask, rightMask in tqdm(zip(listOfImages, listOfLeftMaskImages, listOfRightMaskImages), total=len(listOfImages)):

    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Height, Width))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    leftMask = cv2.imread(leftMask, cv2.IMREAD_GRAYSCALE)
    rightMask = cv2.imread(rightMask, cv2.IMREAD_GRAYSCALE)

    leftMask = cv2.resize(leftMask, (Height, Width))
    rightMask = cv2.resize(rightMask, (Height, Width))
    
    mask = leftMask + rightMask
    mask = cv2.resize(mask, (Height, Width))

    mask[mask > 0] = 1
    maskImages.append(mask)


allImagesNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int)

print("shapes of train and masks :")
print(allImagesNP.shape)
print(maskImagesNP.shape)

from sklearn.model_selection import train_test_split
split = 0.1

train_imgs, valid_imgs = train_test_split(allImagesNP, test_size=split, random_state=42)
train_masks, valid_masks = train_test_split(maskImagesNP, test_size=split, random_state=42)

print("shapes of train and masks :")
print(train_imgs.shape)
print(train_masks.shape)

print("shapes of valid and masks :")
print(valid_imgs.shape)
print(valid_masks.shape)

print("save the train and valid images and masks")
np.save(path + "/train_imgs.npy", train_imgs)
np.save(path + "/train_masks.npy", train_masks)
np.save(path + "/valid_imgs.npy", valid_imgs)
np.save(path + "/valid_masks.npy", valid_masks)

print("done")
