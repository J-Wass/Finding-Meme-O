# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

#load all memes from each folder into nested np array
memes = []
labels = []
for meme_dir in Path('data').iterdir():
    labels.append(str(meme_dir).split('/')[1])
    meme_batch = []
    for meme_img in meme_dir.iterdir():
        meme_batch.append(mpimg.imread(str(meme_img)))
    memes.append(meme_batch)

#create model
classifier = svm.SVC(gamma=0.001)
classifier.fit(memes, labels)
