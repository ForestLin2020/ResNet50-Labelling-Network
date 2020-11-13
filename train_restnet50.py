from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# data path
DATASET_PATH  = 'dataset'

# image size
IMAGE_SIZE = (224, 224)

# class number
NUM_CLASSES = 8

# if lacking GPU memory， you can lower the batch size or reduce the freeze layers
BATCH_SIZE = 8

# freeze layers
FREEZE_LAYERS = 2

# Epoch number
NUM_EPOCHS = 20

# trained model name
WEIGHTS_FINAL = 'model-resnet50-final.h5'


# creating training data and validation data by using througth data augmentation 
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)


valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/test',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)


# index of output class
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# Train the new based on Pretrained ResNet50 model
# Abandon the fully connected layer at the top of ResNet50 in order to building up own label
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
# increace DropOut layer
x = Dropout(0.5)(x)

# increace Dense layer, and using softmax to create the probability of each output
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# setting the frozen layers and training layers
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True


# using the Adam optimizer, and using the lower learning rate to fine-tuning the model
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# output the whole network structure
print(net_final.summary())

# training the model
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

# saving the trained model
net_final.save(WEIGHTS_FINAL)