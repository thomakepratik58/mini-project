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
