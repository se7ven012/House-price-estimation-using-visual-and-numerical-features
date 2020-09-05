#%%
from dataprocessing import datasets
from dataprocessing import models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import argparse
import locale
import os
# for cuDNN debug
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# for cuDNN debug

#%%
# data processing
inputPath = 'data'
df = datasets.load_house_attributes(inputPath+'/HousesInfo.txt')

print("loading images...")
images = datasets.load_house_images(df,inputPath)
images = images / 255.0

print("processing data...")
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# scale  house prices to the range [0, 1] 
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice
 
# min-max scaling on continuous features, one-hot encoding on categorical features
(trainAttrX, testAttrX) = datasets.process_house_attributes(df, trainAttrX, testAttrX)

if not os.path.exists('models'):
    os.makedirs('models')

#%%
# create models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.createResNetV1(64, 64, 3, regress=False)
combinedInput = concatenate([mlp.output, cnn.output])
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
dynamicLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, mode='auto', min_lr=1e-6)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

filename = 'models/model.h5'
model.save(filename)
print('>Saved %s' % filename)

print("training model...")
model.fit(
    [trainAttrX, trainImagesX], trainY,
    validation_data=([testAttrX, testImagesX], testY),
    epochs=200, batch_size=8,callbacks=[dynamicLR])

# accuracy calculation
pred = model.predict([testAttrX, testImagesX])
diff = pred.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("mean: {:.2f}%, std: {:.2f}%".format(mean, std))

#%%
plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)