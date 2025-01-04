#base#@Author: Federico Candela & Maurizio Campolo
#Description: Image Similarity
#09/02/2022 UniRc
# USAGE
# python train.py --dataset data --model model/activity.model --label-bin model/lb.pickle --epochs 50

# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to the input dataset")
ap.add_argument("-m", "--model", required=True,
                help="Path to save the output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
                help="Path to save the output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
                help="Number of epochs to train the network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="Path to save the output loss/accuracy plot")
args = vars(ap.parse_args())

# Define the set of labels in the dataset
LABELS = set(["DOCUMENTARY", "RELIGIOUS_EVENTS", "GAME_SHOW", "TALK_SHOW", "SHOPPING"])

# Load image paths from the dataset directory, initialize data and labels lists
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Process each image in the dataset
for imagePath in imagePaths:
    # Extract the class label from the folder name
    print(imagePath)
    label = imagePath.split(os.path.sep)[-2]

    # Ignore images whose labels are not in the defined set
    if label not in LABELS:
        continue

    # Load the image, convert to RGB, and resize to 224x224
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Append the image and label to the respective lists
    data.append(image)
    labels.append(label)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split the data into training and testing sets (75% training, 25% testing)
trainX, testX, trainY, testY = train_test_split(data, labels, 
                                                test_size=0.25, 
                                                stratify=labels, 
                                                random_state=42)

# Initialize data augmentation for training
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Initialize data augmentation for validation/testing
valAug = ImageDataGenerator()

# Define ImageNet mean subtraction values and apply to data augmentation
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# Load ResNet50 without the top layers
baseModel = ResNet50(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))

# Construct the fully connected (FC) head for the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# Combine the base model and the FC head
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze all layers in the base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the head of the network
print("[INFO] training head...")
H = model.fit(
    x=trainAug.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX) // 32,
    validation_data=valAug.flow(testX, testY),
    validation_steps=len(testX) // 32,
    epochs=args["epochs"]
)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), 
                            target_names=lb.classes_))

# Plot training loss and accuracy
print("[INFO] saving training plot...")
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Serialize the model to disk
print("[INFO] saving trained model...")
model.save(args["model"], save_format="h5")

# Serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
with open(args["label_bin"], "wb") as f:
    f.write(pickle.dumps(lb))
