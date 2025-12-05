# # Builds a transfer-learning model using vgg19 for pneumonia classification.
#     Args:
#         input_shape: Input image dimensions.
#         num_classes: Number of output classes (2 for Normal/Pneumonia).
#         learning_rate: Initial learning rate.
#         fine_tune: If True, unfreezes top layers for fine-tuning.
#     """
import tensorflow as tf
from keras import layers, models    
from keras.applications import VGG19
from keras.optimizers import Adam

def build_vgg19_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001, fine_tune=False):
    # Load the VGG19 model pre-trained on ImageNet, excluding the top layers
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model initially
    base_model.trainable = False

    # Create the model
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # If fine-tuning is enabled, unfreeze the top layers of the base model
    if fine_tune:
        base_model.trainable = True
        # Optionally, you can freeze some layers if needed
        for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4
            layer.trainable = False

        # Recompile the model after making changes to trainable layers
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model