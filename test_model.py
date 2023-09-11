import tensorflow as tf
import os
import cv2
import numpy as np
from keras.preprocessing import image

# list with the classes for the image classification
classes = ["head", "hardhat"]
class_labels = {classes: i for i, classes in enumerate(classes)}
number_of_classes = len(classes)
IMAGE_SIZE = (160, 160)

# load a local model from the saved_models directory
#model = tf.keras.models.load_model('saved_models/mobilenetv2.h5')
model = tf.keras.models.load_model('saved_models/mobilenetv2')
#model.summary()

# test the model by giving it an image and get its prediction
test_image_head_path = "datasets/dataset_test/-.27iko294.ingestion-c484b64b9-xcl5w.jpg"
test_image_head = cv2.imread(test_image_head_path)
test_image_head = cv2.cvtColor(test_image_head, cv2.COLOR_BGR2RGB)
test_image_head = cv2.resize(test_image_head, IMAGE_SIZE)

test_image_hardhat_path = "datasets/dataset_test/-.27ikobka.ingestion-c484b64b9-gnk8v.jpg"
test_image_hardhat = cv2.imread(test_image_hardhat_path)
test_image_hardhat = cv2.cvtColor(test_image_hardhat, cv2.COLOR_BGR2RGB)
test_image_hardhat = cv2.resize(test_image_hardhat, IMAGE_SIZE)

print("Testing the model on an image.....")
# chose either test_image_head or test_image_hardhat for the prediction
image_test_result = model.predict(np.expand_dims(test_image_hardhat, axis=0))
# print the prediction scores/confidence for each class
# index 0 = head, index 1 = hardhat
print(image_test_result[0])

# an easy trick to see the model's prediction scores ("confidence") for each class
# we can get the highest score/confidence among all classes
# map the highest score's index to its class
highest_prediction_score = max(image_test_result[0])
highest_prediction_score_index = 0
for i in range(len(image_test_result[0])):
    if image_test_result[0][i] == highest_prediction_score:
        highest_prediction_score_index = i

most_confident_class = classes[highest_prediction_score_index]
print("The model mostly predicted %s with a score/confidence of %s" %(most_confident_class, highest_prediction_score))

""" Some results after testing the model with a head's image:
Testing the model on an image.....
1/1 [==============================] - 3s 3s/step
[9.992312e-01 7.688053e-04]
The model mostly predicted head with a score/confidence of 0.9992312
"""

""" Some results after testing the model with a hardhat's image:
Testing the model on an image.....
1/1 [==============================] - 2s 2s/step
[0.00373875 0.9962612 ]
The model mostly predicted hardhat with a score/confidence of 0.9962612
"""