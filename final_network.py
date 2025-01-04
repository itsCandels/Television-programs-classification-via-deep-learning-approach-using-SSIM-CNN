#base#@Author: Federico Candela & Maurizio Campolo
#Description: Image Similarity
#09/02/2022 UniRc

# IMPORTS
from pathlib import Path
from datetime import datetime as dt
from skimage.metrics import structural_similarity
import os
import cv2
import time
import numpy as np
import pandas as pd
import argparse

from tensorflow.keras.models import load_model
import pickle
from collections import deque
from csv import reader
import csv

# PROCESSING TIME CALCULATION
start = time.time()

# USER PATHS
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
                help="Path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
                help="Path to label binarizer")
ap.add_argument("-s", "--size", type=int, default=128,
                help="Size of the queue for averaging")

args = vars(ap.parse_args())

# Load the trained model and label binarizer
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# Function to create the mask
def mask_function(imageMath):
    y = 15
    for i in imageMath:
        mask = np.zeros([imageMath.shape[0] - 2 * y, imageMath.shape[1], 3], np.uint8)
        mask = imageMath[y:imageMath.shape[0] - y, 0:imageMath.shape[1]]
        return mask

# Relative paths
db = './DISCRIMINATORI_LAC_RESIZE/'
results_dir = './RESULTS/'
example_clips_dir = './Example_Clips/'

# Variable declarations
threshold = 0.99
threshold = float(threshold)
A3D = []
pathlist = Path(db).glob('**/*.png')
listImage = []

# Process images
for path in pathlist:
    path_in_str = str(path)
    d = os.path.normpath(path_in_str)
    listImage.append(d)
print('Number of images:', len(listImage))

for i in range(len(listImage)):
    gray = cv2.cvtColor(cv2.imread(listImage[i]), cv2.COLOR_BGR2GRAY)
    maskimage = mask_function(gray)
    A3D.append(maskimage)

# Process videos
list_E = []
list_Prova = []
list1 = []
list2 = []
list3 = []
count = 0
processed_suffix = "PROCESSED"
pathvideo = Path(example_clips_dir).glob('**/*.mp4')
data_dir_list = list(pathvideo)
df = pd.DataFrame(data_dir_list)
df.rename(columns={0: 'pathVideo'}, inplace=True)

if os.path.exists('VIDEO_PATHS.csv'):
    for i in data_dir_list:
        k = os.path.abspath(i)
        f = k[:-4]
        list1.append(f)

    with open('VIDEO_PATHS.csv') as reader_obj:
        csv_reader = reader(reader_obj)
        for row in csv_reader:
            for j in row:
                w = os.path.abspath(j)
                nn = w[:-13]
                list2.append(nn)
        extension = os.path.splitext(w)[1]
        new_files = [x + extension for x in list1 + list2 if x not in list2]
        df = pd.DataFrame(new_files)

        if len(new_files) == 0:
            print('No new files')
        else:
            print('New files found')
            df.to_csv('VIDEO_PATHS.csv', mode='a', header=False, index=False)
else:
    df.to_csv('VIDEO_PATHS.csv', mode='a', header=False, index=False)

with open('VIDEO_PATHS.csv') as fileObject:
    reader_obj = csv.reader(fileObject)
    for row in reader_obj:
        for i in row:
            a = os.path.basename(i)
            d = os.path.splitext(a)[1]
            exists = os.path.exists(i)

            if processed_suffix not in i:
                if exists:
                    list_E.append(i)
                    u = i[:-4] + processed_suffix + d
                    list_Prova.append(u)
                    prova = pd.DataFrame(list_Prova)
                    prova.rename(columns={0: 'pathVideo'}, inplace=True)

        for video_path in list_E:
            video_name = os.path.basename(video_path)
            print(video_name)
            cap = cv2.VideoCapture(video_path)
            writer = None
            (W, H) = (None, None)
            found = 0
            frame_list = []
            frame_count = 0
            change_label = None

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            remaining_frames = total_frames - fps

            print("FPS =", fps)
            print("Remaining Frames =", remaining_frames)
            print("Video Path =", cap)
            now = dt.now()
            now_time = now.strftime("%m_%d_%Y_%H:%M:%S")
            print("Date and Time:", now_time)
            frame_results = []

            while frame_count < remaining_frames:
                ret, frame = cap.read()
                frame_count += 1
                if not ret:
                    break
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                if (frame_count % (fps / 2) == 0):
                    movie_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    maskframe = mask_function(movie_frame)

                    frame = cv2.resize(frame, (224, 224)).astype("float32")
                    frame -= mean

                    similarity_score = float(structural_similarity(A3D[i], maskframe))
                    if similarity_score > threshold:
                        preds = model.predict(np.expand_dims(frame, axis=0))[0]
                        Q.append(preds)
                        results = np.array(Q).mean(axis=0)
                        label = lb.classes_[np.argmax(results)]

                        print(f"SIMILARITY_INDEX: {similarity_score} LABEL:{listImage[i].split('/')[-1][:-4]} TIMESTAMP: {now_time} FRAME_COUNT: {frame_count}")
                        frame_results.append(f"{similarity_score} {listImage[i].split('/')[-1][:-4]} {now_time} {label}")

            results_df = pd.DataFrame(frame_results)
            results_df.to_csv(f"{results_dir}{count}_{video_name}.csv")
            count += 1
            cap.release()

# END OF PROCESSING TIME CALCULATION
elapsed = time.time() - start
output = dt.strftime(dt.utcfromtimestamp(elapsed), '%H:%M:%S')
print("PROCESSING TIME:", output)
