import shap
import numpy as np
from keras.preprocessing import image
import os, matplotlib.pyplot as plt

def shap_image_explain(model, img_path, output_path='results/xai_outputs'):
    os.makedirs(output_path, exist_ok=True)

    # Load & preprocess image
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    x = image.img_to_array(img)[np.newaxis, ...] / 255.0

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    # Baseline: black image (simulate normal lungs)
    background = np.zeros((1, 224, 224, 3))

    masker = shap.maskers.Image("blur(64,64)", (224, 224, 3))
    explainer = shap.Explainer(model, masker, algorithm="partition")
    shap_values = explainer(x)

    shap_values_arr = shap_values.values if hasattr(shap_values, "values") else shap_values
    if isinstance(shap_values_arr, list):
        shap_values_arr = shap_values_arr[0]

    # Create overlay visualization
    shap_values_abs = np.abs(shap_values_arr[0]).sum(-1)
    shap_values_abs = shap_values_abs / np.max(shap_values_abs)

    plt.imshow(x[0])
    plt.imshow(shap_values_abs, cmap="jet", alpha=0.5)
    plt.axis("off")

    out_path = os.path.join(output_path, os.path.basename(img_path).split('.')[0] + '_shap_fixed.png')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"SHAP explanation saved to {out_path}")
    return out_path
