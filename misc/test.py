import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import os

test_dir = "/Users/naba/Desktop/freelance/data_qr/test"
save_path = "/Users/naba/Desktop/freelance/data_qr/saved_models"

os.makedirs(save_path, exist_ok=True)

img_size = (128, 128)
batch_size = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

best_model = tf.keras.models.load_model(os.path.join(save_path, "qr_model.h5"))

# Evaluate on test set
results = best_model.evaluate(test_ds, return_dict=True)
print("\nTest Metrics (Best Model):")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Compute F1 score manually from precision & recall
precision = results['precision']
recall = results['recall']
f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
print(f"F1 Score: {f1:.4f}")

# Compute MAP (Mean Average Precision)
y_true, y_pred = [], []
for x, y in test_ds:
    preds = best_model.predict(x, verbose=0)
    y_true.extend(y.numpy().flatten())
    y_pred.extend(preds.flatten())

ap = tf.keras.metrics.AUC(curve="PR")(y_true, y_pred).numpy()
print(f"Mean Average Precision (MAP): {ap:.4f}")
