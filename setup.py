# setup_project.py
import os
import textwrap
import json

project_root = os.getcwd()  # run from project root

# --- Directory structure to create ---
dirs = [
    "data",  # NOTE: your data/chest_xray should already exist
    "models/checkpoints",
    "src/explainability",
    "notebooks",
    "results/gradcam_heatmap_examples",
    "results/shap_explanations",
    "results/lime_explanations",
    "logs",
    "scripts",
    "docs"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# --- Files and their boilerplate content ---
files = {}

# README
files["README.md"] = textwrap.dedent("""\
    # Pneumonia-XAI
    ResNet50V2-based pneumonia detection (transfer learning) + explainability (Grad-CAM, SHAP, LIME).
    
    Project structure created by setup_project.py.
    
    Usage (after installing requirements):
    
    1. Place your dataset at: data/chest_xray/  (it should contain train/, val/, test/)
    2. Train the model:
       python train.py
    3. Evaluate the model:
       python evaluate.py
    4. After training, run explainability notebooks or scripts in src/explainability.
    """)

# config.json
config = {
    "IMG_HEIGHT": 224,
    "IMG_WIDTH": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "TRAIN_DIR": "data/chest_xray/train",
    "VAL_DIR": "data/chest_xray/val",
    "TEST_DIR": "data/chest_xray/test",
    "MODEL_SAVE_PATH": "models/trained_model.h5",
    "CHECKPOINT_DIR": "models/checkpoints",
    "RESULTS_DIR": "results"
}
files["config.json"] = json.dumps(config, indent=4)

# requirements.txt
files["requirements.txt"] = textwrap.dedent("""\
    tensorflow==2.16.1
    keras==3.3.3
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    opencv-python
    shap==0.45.0
    lime==0.2.0.1
    pillow
    """)

# models/resnet50v2_pneumonia_model.py
files["models/resnet50v2_pneumonia_model.py"] = textwrap.dedent("""\
    from keras.applications import ResNet50V2
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, AveragePooling2D
    from keras.optimizers import Adam

    def CreateModel(img_height=224, img_width=224):
        \"\"\"Builds a ResNet50V2-based Sequential model for binary classification.\"\"\"
        base_model = ResNet50V2(include_top=False, input_shape=(img_height, img_width, 3), weights='imagenet')
        base_model.trainable = False  # freeze base for transfer learning

        model = Sequential([
            base_model,
            AveragePooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=3.5e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        return model
    """)

# src/data_utils.py
files["src/data_utils.py"] = textwrap.dedent("""\
    import json
    from keras.preprocessing.image import ImageDataGenerator

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
        return train, val, test
    """)

# src/train_model.py (core training logic)
files["src/train_model.py"] = textwrap.dedent("""\
    import os
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from models.resnet50v2_pneumonia_model import CreateModel
    from src.data_utils import get_data_generators, load_config

    conf = load_config()

    train_dir = conf['TRAIN_DIR']
    val_dir = conf['VAL_DIR']
    test_dir = conf['TEST_DIR']
    IMG = (conf['IMG_HEIGHT'], conf['IMG_WIDTH'])
    BATCH = conf['BATCH_SIZE']
    EPOCHS = conf['EPOCHS']

    model = CreateModel(img_height=IMG[0], img_width=IMG[1])

    train, val, test = get_data_generators(train_dir, val_dir, test_dir, img_size=IMG, batch_size=BATCH)

    os.makedirs(conf['CHECKPOINT_DIR'], exist_ok=True)
    checkpoint_path = os.path.join(conf['CHECKPOINT_DIR'], 'best_model.keras')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_sparse_categorical_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train,
        validation_data=val,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(conf['MODEL_SAVE_PATH'])
    print(f\"Model trained and saved to {conf['MODEL_SAVE_PATH']}\")
    """)

# src/evaluate_model.py
files["src/evaluate_model.py"] = textwrap.dedent("""\
    from keras.models import load_model
    from src.data_utils import get_data_generators, load_config
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    conf = load_config()
    IMG = (conf['IMG_HEIGHT'], conf['IMG_WIDTH'])
    BATCH = conf['BATCH_SIZE']

    model = load_model(conf['MODEL_SAVE_PATH'])
    _, _, test = get_data_generators(conf['TRAIN_DIR'], conf['VAL_DIR'], conf['TEST_DIR'], img_size=IMG, batch_size=BATCH)

    preds = model.predict(test)
    y_pred = np.argmax(preds, axis=1)
    y_true = test.classes

    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL','PNEUMONIA'], yticklabels=['NORMAL','PNEUMONIA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    os.makedirs(conf['RESULTS_DIR'], exist_ok=True)
    plt.savefig(os.path.join(conf['RESULTS_DIR'], 'confusion_matrix.png'))
    print('Saved confusion matrix to results/confusion_matrix.png')
    """)

# train.py (top-level launcher)
files["train.py"] = textwrap.dedent("""\
    # Top-level training launcher
    from src.train_model import *
    # Running src/train_model.py will execute training when invoked via "python train.py"
    print('Training script executed. Check terminal for training logs.')
    """)

# evaluate.py (top-level)
files["evaluate.py"] = textwrap.dedent("""\
    # Top-level evaluation launcher
    from src.evaluate_model import *
    print('Evaluation script executed. Check results/ for outputs.')
    """)

# explainability placeholders
files["src/explainability/gradcam_utils.py"] = textwrap.dedent("""\
    import numpy as np
    import cv2
    import tensorflow as tf
    from keras.preprocessing import image
    import matplotlib.pyplot as plt
    import os

    def gradcam_heatmap(model, img_path, last_conv_layer_name=None, output_path='results/gradcam_heatmap_examples'):
        os.makedirs(output_path, exist_ok=True)
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Determine layer name automatically if not provided
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name:
                    last_conv_layer_name = layer.name
                    break

        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = np.zeros(conv_outputs.shape[0:2])
        for i in range(pooled_grads.shape[-1]):
            heatmap += pooled_grads[i] * conv_outputs[:,:,i]
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        orig = cv2.imread(img_path)
        orig = cv2.resize(orig, (224,224))
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        save_path = os.path.join(output_path, os.path.basename(img_path).split('.')[0] + '_gradcam.jpg')
        cv2.imwrite(save_path, overlay)
        return save_path
    """)

files["src/explainability/shap_utils.py"] = textwrap.dedent("""\
    import shap
    import numpy as np
    import os
    from keras.preprocessing import image

    def shap_image_explain(model, img_path, background_samples=10, output_path='results/shap_explanations'):
        os.makedirs(output_path, exist_ok=True)
        # Load image and preprocess (model expects [0,1] scaled 224x224)
        from keras.preprocessing import image as kimage
        img = kimage.load_img(img_path, target_size=(224,224))
        x = kimage.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # build background
        background = np.random.randn(background_samples, 224, 224, 3)
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(x)
        # shap.image_plot expects a particular shape order, this is a placeholder
        shap.image_plot(shap_values, -x)
        # Save logic should be added if image_plot used interactively
        return True
    """)

files["src/explainability/lime_utils.py"] = textwrap.dedent("""\
    import numpy as np
    import os
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from keras.preprocessing import image as kimage
    import cv2

    def lime_image_explain(model, img_path, output_path='results/lime_explanations'):
        os.makedirs(output_path, exist_ok=True)
        img = kimage.load_img(img_path, target_size=(224,224))
        x = kimage.img_to_array(img).astype('double') / 255.0

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(x, model.predict, top_labels=2, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        lime_visual = mark_boundaries(temp/255.0, mask)
        out_path = os.path.join(output_path, os.path.basename(img_path).split('.')[0] + '_lime.png')
        cv2.imwrite(out_path, (lime_visual * 255).astype('uint8'))
        return out_path
    """)

# notebooks placeholders
files["notebooks/01_preprocessing.ipynb"] = textwrap.dedent("""\
    {
     "cells": [],
     "metadata": {},
     "nbformat": 4,
     "nbformat_minor": 5
    }
    """)
files["notebooks/02_training.ipynb"] = files["notebooks/01_preprocessing.ipynb"]
files["notebooks/03_evaluation.ipynb"] = files["notebooks/01_preprocessing.ipynb"]
files["notebooks/04_explainability.ipynb"] = files["notebooks/01_preprocessing.ipynb"]

# docs/LICENSE (placeholder)
files["docs/LICENSE"] = "MIT License - placeholder"

# scripts: helpful shell scripts
files["scripts/run_train.sh"] = textwrap.dedent("""\
    #!/bin/bash
    python train.py
    """)
files["scripts/run_eval.sh"] = textwrap.dedent("""\
    #!/bin/bash
    python evaluate.py
    """)

# top-level utility: small helper to show status
files["status.txt"] = "Project scaffold created by setup_project.py"

# write all files
for path, content in files.items():
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Full project scaffold created.")
print()
print("Next steps:")
print("1) Put your existing 'chest_xray' folder inside ./data/ so the path is: data/chest_xray/")
print("2) Create a virtual environment and install dependencies:")
print("   python -m venv venv")
print("   source venv/bin/activate   # on Linux/Mac")
print("   venv\\Scripts\\activate    # on Windows")
print("   pip install -r requirements.txt")
print("3) Train the model:")
print("   python train.py")
print("4) Evaluate:")
print("   python evaluate.py")
print()
print("If you want, I can now generate the explainability scripts with more robust saving and error-checking (Grad-CAM / SHAP / LIME).")
