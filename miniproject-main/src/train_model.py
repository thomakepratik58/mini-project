import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from models.resnet50v2_pneumonia_model import CreateModel
# from models.efficientnetv2_model import build_efficientnetv2_model
from models.vgg19_model import build_vgg19_model
from src.data_utils import get_data_generators, load_config

conf = load_config()

train_dir = conf['TRAIN_DIR']
val_dir = conf['VAL_DIR']
test_dir = conf['TEST_DIR']
IMG = (conf['IMG_HEIGHT'], conf['IMG_WIDTH'])
BATCH = conf['BATCH_SIZE']
EPOCHS = conf['EPOCHS']

model = build_vgg19_model(input_shape=(IMG[0], IMG[1], 3))


train, val, test = get_data_generators(train_dir, val_dir, test_dir, img_size=IMG, batch_size=BATCH)

os.makedirs(conf['CHECKPOINT_DIR'], exist_ok=True)
checkpoint_path = os.path.join(conf['CHECKPOINT_DIR'], 'best_model.keras')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_sparse_categorical_accuracy',
        mode='max',                     # explicitly maximize accuracy
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]


history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=callbacks
)
# final accuracy of overall model
test_loss, test_acc = model.evaluate(test)
print(f"Test accuracy: {test_acc:.4f}")

model.save(conf['MODEL_SAVE_PATH'])
print(f"Model trained and saved to {conf['MODEL_SAVE_PATH']}")
