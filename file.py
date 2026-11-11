import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.metrics import roc_curve, auc

MODEL_PATH = "malaria_model.keras"

# =========================
# CONFIG
# =========================
DATA_DIR = "./cell_images"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)
# =========================
# CLASS WEIGHTS
# =========================
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN OR LOAD
# =========================
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)

else:
    print("Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )

    model.save(MODEL_PATH)
    print("Model saved!")

# =========================
# EVALUATION
# =========================
print("\n--- Evaluating Model ---")

val_gen.reset()
preds = model.predict(val_gen)
preds_binary = (preds > 0.5).astype(int)

class_names = ['INFECTED', 'UNINFECTED']

print(classification_report(
    val_gen.classes,
    preds_binary,
    target_names=class_names
))

cm = confusion_matrix(val_gen.classes, preds_binary)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
# =========================
# ROC CURVE
# =========================
fpr, tpr, thresholds = roc_curve(val_gen.classes, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

plt.grid()
plt.show()


def predict_image(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.expand_dims(img_rgb / 255.0, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "INFECTED" if pred < 0.5 else "UNINFECTED"
    confidence = 1 - pred if pred < 0.5 else pred
    print("\n===== RESULT =====")
    print("Raw Output :", round(pred, 3))
    print("Prediction :", label)
    print("Confidence :", round(confidence, 3))

# =========================
# RUN TEST
# =========================
my_test_image = "./uninfected.png"
predict_image(my_test_image, model)
