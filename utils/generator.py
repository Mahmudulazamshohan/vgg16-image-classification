from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1. / 255) 
def create_train_generator(train_dir,batch_size,img_width=224,img_height=224):
   return datagen.flow_from_directory(train_dir,
   target_size=(img_width, img_height),
   batch_size=batch_size,
   class_mode=None,
   shuffle=False) 
def create_validation_generator(validation_dir,batch_size,img_width=224,img_height=224):
   return datagen.flow_from_directory(validation_dir,
   target_size=(img_width, img_height),
   batch_size=batch_size,
   class_mode=None,
   shuffle=False)