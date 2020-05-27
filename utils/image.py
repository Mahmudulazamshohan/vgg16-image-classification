from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
import numpy as np
def read_image(file_path,image_size=(224,224)):
   print("[INFO] loading and preprocessing image…") 
   image = load_img(file_path, target_size=image_size) 
   image = img_to_array(image) 
   image = np.expand_dims(image, axis=0)
   image /= 255. 
   return image