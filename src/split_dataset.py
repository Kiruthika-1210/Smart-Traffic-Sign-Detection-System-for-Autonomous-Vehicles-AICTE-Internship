import os
import random
import shutil

# INPUT folders
images_path = "../data/images"
labels_path = "../data/labels"

# OUTPUT folders
train_img_out = "../data/images/train"
test_img_out = "../data/images/test"

train_lbl_out = "../data/labels/train"
test_lbl_out = "../data/labels/test"

# Create folders
os.makedirs(train_img_out, exist_ok=True)
os.makedirs(test_img_out, exist_ok=True)
os.makedirs(train_lbl_out, exist_ok=True)
os.makedirs(test_lbl_out, exist_ok=True)

# Collect all image filenames
images = [
    f for f in os.listdir(images_path)
    if f.endswith(".png") or f.endswith(".jpg")
]

# Shuffle for randomness
random.shuffle(images)

# 80/20 split
split_index = int(0.8 * len(images))
train_files = images[:split_index]
test_files = images[split_index:]

# COPY TRAIN FILES
for img in train_files:
    img_src = os.path.join(images_path, img)
    shutil.copy(img_src, train_img_out)

    label = img.rsplit(".", 1)[0] + ".txt"
    lbl_src = os.path.join(labels_path, label)

    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, train_lbl_out)

# COPY TEST FILES
for img in test_files:
    img_src = os.path.join(images_path, img)
    shutil.copy(img_src, test_img_out)

    label = img.rsplit(".", 1)[0] + ".txt"
    lbl_src = os.path.join(labels_path, label)

    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, test_lbl_out)

print("✅ Dataset successfully split into TRAIN and TEST folders!")
