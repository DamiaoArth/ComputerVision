import numpy as np
import os

# Caminho para o diretório de dados
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Load the data
print("Start loading...")
allImagesNp = np.load(os.path.join(data_dir, "train_imgs.npy"))
maskImagesNp = np.load(os.path.join(data_dir, "train_masks.npy"))
allValidateImagesNP = np.load(os.path.join(data_dir, "valid_imgs.npy"))
maskValidateImagesNP = np.load(os.path.join(data_dir, "valid_masks.npy"))

# Print shapes
print(allImagesNp.shape)
print(maskImagesNp.shape)
print(allValidateImagesNP.shape)
print(maskValidateImagesNP.shape)

# Set height and width
Height = 256
Width = 256

# Build the model
import tensorflow as tf
from stp2unet import build_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (Height, Width, 3)  # Define input shape
lr = 1e-4  # Learning rate
batchSize = 4
epochs = 50

# Build the model
model = build_model(shape)

# Print model summary
print(model.summary())

opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

stepsPerEpoch = np.ceil(len(allImagesNp) / batchSize)
validationSteps = np.ceil(len(allValidateImagesNP) / batchSize)

best_model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "lungUnet.h5")

# Callbacks
callbacks = [
    ModelCheckpoint(filepath=best_model_file, verbose=1 , save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor="val_loss", patience=20, verbose=1)
]

history = model.fit(allImagesNp, maskImagesNp, batch_size=batchSize, epochs=epochs, verbose=1,
                    validation_data=(allValidateImagesNP, maskValidateImagesNP),
                    validation_steps=validationSteps,
                    steps_per_epoch=stepsPerEpoch,
                    shuffle=True,
                    callbacks=callbacks)
