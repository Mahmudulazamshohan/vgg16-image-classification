from utils.image import read_image
from utils.generator import create_train_generator
from layers.model import create_tf_model
from keras import applications
import time
path = 'rose-peach.jpg'
images = read_image(path)
model=create_tf_model(5)
train_generator = create_train_generator('data/train',50)
image_classes=[image_class for image_class in train_generator.class_indices.keys()]
model.load_weights('weights/trained_model.h5')
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
time.sleep(.5)
bt_prediction = vgg16.predict(images) 
preds = model.predict_proba(bt_prediction)
for idx, label, x in zip(range(0,6), image_classes , preds[0]):
  print("Index: {}, Label: {} {}%".format(idx, label, round(x*100,2) ))
