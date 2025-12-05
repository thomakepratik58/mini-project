import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from scipy.stats import pearsonr
from keras.models import load_model
from src.data_utils import get_data_generators, load_config
from src.explainability.gradcam_utils import gradcam_heatmap
from src.explainability.lime_utils import lime_image_explain
from src.explainability.shap_utils import shap_image_explain


# CONFIGURATION

conf = load_config()
MODEL_PATH = "models/trained_model.h5"  
OUTPUT_DIR = "results/xai_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD MODEL

print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# LOAD TEST DATA

_, _, test_gen = get_data_generators(
    conf['TRAIN_DIR'], conf['VAL_DIR'], conf['TEST_DIR'],
    img_size=(conf['IMG_HEIGHT'], conf['IMG_WIDTH']),
    batch_size=conf['BATCH_SIZE']
)


# GET MODEL PREDICTIONS

print("Running model predictions...")
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)


# CONFUSION MATRIX

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved.")

# 
#  ROC CURVE
fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:,1])
auc = roc_auc_score(y_true, y_pred_probs[:,1])
plt.figure()
plt.plot(fpr, tpr, label=f'AI model (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.title('ROC Curve')
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), bbox_inches='tight')
plt.close()
print("ROC curve saved.")


# CORRELATION + BLAND–ALTMAN (simulated severity)

ai_scores = y_pred_probs[:,1] * 40
radiologist_scores = np.random.normal(ai_scores, 5)

# Correlation
r, _ = pearsonr(ai_scores, radiologist_scores)
plt.figure()
plt.scatter(radiologist_scores, ai_scores, alpha=0.6)
plt.plot([0,40],[0,40],'r--')
plt.xlabel("Radiologist Severity Index")
plt.ylabel("AI Severity Index")
plt.title(f"Correlation (PCC={r:.2f})")
plt.savefig(os.path.join(OUTPUT_DIR, "severity_correlation.png"), bbox_inches='tight')
plt.close()

# Bland–Altman
diff = ai_scores - radiologist_scores
mean = (ai_scores + radiologist_scores) / 2
plt.figure()
plt.scatter(mean, diff, alpha=0.6)
plt.axhline(np.mean(diff), color='red')
plt.axhline(np.mean(diff)+1.96*np.std(diff), color='gray', linestyle='--')
plt.axhline(np.mean(diff)-1.96*np.std(diff), color='gray', linestyle='--')
plt.xlabel("Average Severity Index")
plt.ylabel("Difference (AI - Radiologist)")
plt.title("Bland–Altman Agreement")
plt.savefig(os.path.join(OUTPUT_DIR, "bland_altman.png"), bbox_inches='tight')
plt.close()
print("Correlation and Bland–Altman saved.")

    
# RUN XAI ON 2 TEST IMAGES
test_images = test_gen.filepaths[:2]
print(f"Generating explanations for {len(test_images)} test images...")

for idx, img_path in enumerate(test_images):
    print(f"\nProcessing {os.path.basename(img_path)} ({idx+1}/{len(test_images)})...")

    gradcam_out = gradcam_heatmap(model, img_path, output_path=os.path.join(OUTPUT_DIR, f"gradcam_{idx+1}.png"))
    lime_out = lime_image_explain(model, img_path, output_path=OUTPUT_DIR)
    shap_out = shap_image_explain(model, img_path, output_path=OUTPUT_DIR)

    # Lesion Overlay
    ai_heatmap = cv2.imread(gradcam_out)
    gray_cxr = cv2.imread(img_path)
    gray_cxr = cv2.resize(gray_cxr, (ai_heatmap.shape[1], ai_heatmap.shape[0]))
    overlay = cv2.addWeighted(gray_cxr, 0.6, ai_heatmap, 0.4, 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"lesion_overlay_{idx+1}.png"), overlay)

print("\n All visualizations complete!")

