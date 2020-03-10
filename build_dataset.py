from cnn import config 
from imutils import paths 
import random
import shutil
import os

imagePaths = list(paths.list_images(config.orig_input_dataset))
random.seed(42)
random.shuffle(imagePaths)

i = int(len(imagePaths) * config.train_split)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

i = int(len(trainPaths) * config.val_split)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]


datasets = [
	("training", trainPaths, config.train_path),
	("validation", valPaths, config.val_path),
	("testing", testPaths, config.test_path)
    ]

for (dType, imagePaths, baseOutput) in datasets:
    print("[INFO] building '{}' split".format(dType))
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)
    for inputPath in imagePaths:
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]
        labelPath = os.path.sep.join([baseOutput, label])
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
            
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)