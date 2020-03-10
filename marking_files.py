import glob
import os
path = 'C:/Users/Dell GTX/Desktop/Glaucoma_img/name'
files = os.listdir(path)
i = 1

for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.png'])))