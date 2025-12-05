import json
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def load_config(path='config.json'):
    import json
    with open(path,'r') as f:
        return json.load(f)

def get_data_generators(train_dir, val_dir, test_dir,
                        img_size=(224,224), batch_size=32):
    # augmentation for training
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.1
    )
    val_test_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=True
    )
    val = val_test_gen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )
    test = val_test_gen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )
    print(json.dumps(train.class_indices, indent=4))
    return train, val, test
