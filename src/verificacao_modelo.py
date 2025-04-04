from keras.models import load_model

# Carregar o modelo salvo
model_path = "keras_model.h5"
model = load_model(model_path)

# Exibir a estrutura do modelo
model.summary()

