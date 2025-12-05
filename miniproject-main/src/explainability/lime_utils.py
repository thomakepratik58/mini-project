import numpy as np
import os, cv2
from lime import lime_image
from skimage.segmentation import felzenszwalb, mark_boundaries
from keras.preprocessing import image as kimage
import matplotlib.pyplot as plt

def lime_image_explain(model, img_path, output_path='results/xai_outputs', n_segments=10):
    os.makedirs(output_path, exist_ok=True)

    # Load and preprocess image
    img = kimage.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    x = kimage.img_to_array(img).astype('double') / 255.0
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    # Custom segmentation function tuned for CXR
    def segment_fn(image):
        return felzenszwalb(image, scale=150, sigma=0.6, min_size=100)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        x,
        model.predict,
        segmentation_fn=segment_fn,
        top_labels=1,
        num_samples=1500
    )

    # Get mask and explanation overlay
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=n_segments,
        hide_rest=False
    )

    # Blend explanation with original image
    blended = mark_boundaries(temp / 255.0, mask)
    out_path = os.path.join(output_path, os.path.basename(img_path).split('.')[0] + '_lime_fixed.png')

    plt.imshow(blended)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"LIME explanation saved to {out_path}")
    return out_path
