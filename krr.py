import numpy as np
import cv2
from sklearn.kernel_ridge import KernelRidge

train = np.loadtxt("train.txt", dtype=str)
valid = np.loadtxt("validation.txt", dtype=str)
test = np.loadtxt("test.txt", dtype=str)


def separate_label(data):
    paths = []
    images = []
    labels = []
    for row in data[1:]:
        row = row.split(',')
        label = int(row[1])

        if label < 2:
            if label == 0:
                labels.append(-1)
            else:
                labels.append(1)
            paths.append(row[0])

    for path in paths:
        image = cv2.imread("train+validation/" + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.reshape(image, (-1,))
        images.append(image)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


train_data, train_labels = separate_label(train)
valid_data, valid_labels = separate_label(valid)
# test_data = separate_label(test, trainFlag=False)

model = KernelRidge()
model.fit(train_data, train_labels)
predictions = model.predict(valid_data)
predicted_label = []
for prediction in predictions:
    if prediction < 0:
        predicted_label.append(-1)
    else:
        predicted_label.append(1)

score = np.mean(predicted_label == valid_labels)
print("Score: ", score)
