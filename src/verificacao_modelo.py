import os
from tensorflow.keras.models import load_model

# Carregar o modelo salvo
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "unet", "models", "keras_model.h5")
model = load_model(model_path)

# Exibir a estrutura do modelo
model.summary()

