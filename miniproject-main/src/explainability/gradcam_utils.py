import numpy as np
import tensorflow as tf
import cv2, os
from keras.preprocessing import image

def gradcam_heatmap(model, img_path, output_path='results/xai_outputs/gradcam.png', layer_name='conv5_block3_out'):
    """
    Generates Grad-CAM heatmap and overlay for a given model and image.
    Compatible with Sequential models containing ResNet50V2 or similar architectures.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load image
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)[np.newaxis, ...] / 255.0

    # Get base_model if model is Sequential
    if hasattr(model.layers[0], "layers"):
        base_model = model.layers[0]
    else:
        base_model = model

    # Check if specified layer exists
    if layer_name not in [layer.name for layer in base_model.layers]:
        print(f"Layer '{layer_name}' not found. Available layers:")
        for l in base_model.layers[-10:]:
            print(" -", l.name)
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=[base_model.input],
        outputs=[base_model.get_layer(layer_name).output, base_model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else heatmap

    # Convert to NumPy array safely
    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()

    # Resize and overlay heatmap
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(output_path, overlay)
    print(f"Grad-CAM saved to {output_path}")
    return output_path
