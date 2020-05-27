from layers.model import create_tf_model
from utils.dataset import pretrain_dataset,load_labels
import datetime
#Batch Size
batch_size=50
# Vgg16 dataset train dataset
train_data=pretrain_dataset()
# Load labels
train_labels=load_labels()
start = datetime.datetime.now()

# Keras Model with total class
model = create_tf_model(5)
history = model.fit(train_data, train_labels, epochs=7,batch_size=batch_size)
model.save_weights("weights/trained_model.h5")
# Evaluate Model
(eval_loss, eval_accuracy) = model.evaluate(train_data, train_labels, batch_size=batch_size,verbose=1)
print("Accuracy: {:.2f}%".format(eval_accuracy * 100)) 
print("Loss: {}".format(eval_loss)) 
end= datetime.datetime.now()
elapsed= end-start
print ('Total Time:', elapsed)