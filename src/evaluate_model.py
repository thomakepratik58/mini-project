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
