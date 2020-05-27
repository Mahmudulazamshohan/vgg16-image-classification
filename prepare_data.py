import datetime
from keras import applications 
from utils.generator import create_train_generator,create_validation_generator
from utils.dataset import save_dataset,save_labels
import math
from keras.utils.np_utils import to_categorical 
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
start = datetime.datetime.now()
# create batch size of generator
batch_size=50
#train dataset generator
train_generator = create_train_generator('data/train',batch_size)
#validation dataset generator
validation_generator = create_validation_generator('data/test',batch_size)
nb_train_samples = len(train_generator.filenames) 
num_classes = len(train_generator.class_indices) 
predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
bottleneck_train = vgg16.predict_generator(train_generator, predict_size_train) 
save_dataset(bottleneck_train)
save_labels(to_categorical(train_generator.classes,num_classes))
end= datetime.datetime.now()
elapsed= end-start
print ('Total Time Needed: {}'.format(elapsed))