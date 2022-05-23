import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

# importarea clasificatoarelor
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# importari pentru preprocesarea datelor
from skimage import color

# citirea si incarcarea datelor de TRAIN

f = open("train.txt", "r")

line = f.readline()
train_images_names = []
train_images_labels = []


for line in f.readlines():
    line = [x for x in line.split(",")]
    line[1] = int(line[1].rstrip('\n'))
    train_images_names.append(line[0])
    train_images_labels.append(line[1])

train_images = []
train_labels = []
for im in glob.glob("train+validation\*.png"):
    im_name = im.lstrip(r"train+validation\ ")
    if im_name in train_images_names:
        im = cv2.imread(im)
        train_images.append(im.flatten())
        train_labels.append(train_images_labels[train_images_names.index(im_name)])

# preprocesarea datelor de TRAIN in flipped

train_images_flipped = []
for im in glob.glob("train+validation\*.png"):
    im_name = im.lstrip(r"train+validation\ ")
    if im_name in train_images_names:
        im = cv2.imread(im)
        # rotirea imaginii pe verticala
        im = cv2.flip(im, 0)
        train_images_flipped.append(im.flatten())

# citirea si incarcarea datelor de VALIDARE

f = open("validation.txt", "r")

line = f.readline()
validation_images_names = []
validation_images_labels = []

for line in f.readlines():
    line = [x for x in line.split(",")]
    line[1] = int(line[1].rstrip('\n'))
    validation_images_names.append(line[0])
    validation_images_labels.append(line[1])

validation_images = []
validation_labels = []
for im in glob.glob("train+validation\*.png"):
    im_name = im.lstrip(r"train+validation\ ")
    if im_name in validation_images_names:
        im = cv2.imread(im)
        validation_images.append(im.flatten())
        validation_labels.append(validation_images_labels[validation_images_names.index(im_name)])

# preprocesarea datelor de VALIDARE in flippedscale

validation_images_flipped = []
for im in glob.glob("train+validation\*.png"):
    im_name = im.lstrip(r"train+validation\ ")
    if im_name in validation_images_names:
        im = cv2.imread(im)
        im = cv2.flip(im, 0)
        validation_images_flipped.append(im.flatten())

# citirea si incarcarea datelor de TEST

f = open("test.txt", "r")

line = f.readline()
test_images_names = []

for line in f.readlines():
    test_images_names.append(line)

test_images_names.sort(key = str.casefold)

test_images = []
for im in glob.glob("test\*.png"):
    im = cv2.imread(im)
    im = cv2.flip(im, 0)
    test_images.append(im.flatten())

f.close()

# antrenarea cu datele normale si datele flipped

train_images = train_images + train_images_flipped
train_labels = train_labels + train_labels
validation_images = validation_images + validation_images_flipped
validation_labels = validation_labels + validation_labels

# antrenarea doar cu datele flipped

# train_images = train_images_flipped
# validation_images = validation_images_flipped

# clasificator NB

model = MultinomialNB()

model.fit(train_images, train_labels)
validation_predictions = model.predict(validation_images)
print("NB:", end = " ")
print(np.mean(validation_predictions == validation_labels))

# clasificator SVM

model = svm.SVC()

model.fit(train_images, train_labels)
validation_predictions = model.predict(validation_images)
print("SVM:", end = " ")
print(np.mean(validation_predictions == validation_labels))

# clasificator MLP

model = MLPClassifier()

model.fit(train_images, train_labels)
validation_predictions = model.predict(validation_images)
print("MLP:", end = " ")
print(np.mean(validation_predictions == validation_labels))

# test_predictions = model.predict(test_images)

# f = open("submissionSVM4.txt", "w")

# f.write("id,label\n")

# for i in range(len(test_images_names)):
#     string = str(test_images_names[i]).rstrip('\n') + "," + str(test_predictions[i]).rstrip('\n') + "\n"
#     f.write(string)

# f.close()