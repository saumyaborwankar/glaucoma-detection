from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from cnn.resnet import ResNet
from cnn import config
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

import matplotlib
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

num_epochs = 25
init_lr= 1e-1
bs = 32
 


def poly_decay(epoch):

	maxEpochs = num_epochs
	baseLR = init_lr
	power = 0.5 
 
	
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 

	return alpha


totalTrain = len(list(paths.list_images(config.train_path)))
totalVal = len(list(paths.list_images(config.val_path)))
totalTest = len(list(paths.list_images(config.test_path)))

trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
	config.train_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=bs)
 

valGen = valAug.flow_from_directory(
	config.val_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)
 

testGen = valAug.flow_from_directory(
	config.test_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)

model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=init_lr, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // bs,
	validation_data=valGen,
	validation_steps=totalVal // bs,
	epochs=num_epochs,
	callbacks=callbacks)

print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // bs) + 1)
 

predIdxs = np.argmax(predIdxs, axis=1) 
 

print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

N = num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])