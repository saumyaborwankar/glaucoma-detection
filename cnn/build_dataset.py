from cnn import config # our configuration file 
from imutils import paths # which you need to download! 
import random
import shutil
import os

# shuffle all the images in the original input directory 
imagePaths = list(paths.list_images(config.orig_input_dataset))
random.seed(42)
random.shuffle(imagePaths)
# split the data into testing and training 
i = int(len(imagePaths) * config.train_split)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# set aside some of the training data for validation data 
i = int(len(trainPaths) * config.val_split)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# define the training/validation/testing datasets 
datasets = [
	("training", trainPaths, config.train_path),
	("validation", valPaths, config.val_path),
	("testing", testPaths, config.test_path)
    ]
# loop over the datasets
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