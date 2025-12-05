# models/resnet50v2_pneumonia_model.py
from keras.applications import ResNet50V2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.optimizers import Adam

def CreateModel(input_shape=(224, 224, 3), num_classes=2, learning_rate=1e-4, fine_tune=False):
    """
    Builds a transfer-learning model using ResNet50V2 for pneumonia classification.
    Args:
        input_shape: Input image dimensions.
        num_classes: Number of output classes (2 for Normal/Pneumonia).
        learning_rate: Initial learning rate.
        fine_tune: If True, unfreezes top layers for fine-tuning.
    """
    # Load pretrained ResNet50V2 without top classification layer
    base_model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    # Freeze or unfreeze layers
    if not fine_tune:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in base_model.layers[:-20]:  # keep last ~20 layers trainable
            layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
