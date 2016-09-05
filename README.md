1) data_prepare.py
Must be run first. Reads the training and test images, pads them to make them square and resizes them to 96x96. Writes the resulting images in numpy arrays.


2) train_cnn.py
Must be run second. Trains a relatively small convolutional neural network from scratch on all the train images. Saves the weights based on average train accuracy and early stops. A lot of data augmentation is being employed including taking advantage of the rot90 symmetry between classes.


3) predict.py
Makes the predictions based on the trained model. It gets the probabilities for each of the 4 categories from the CNN for various transformations of each image (rot90 + flips) and stacks them
as meta features. A Random Forest is being trained and use to make the final predictions on these meta features.

4)models.py
The definition of the CNN 


5)utils.py
Definitions from the data augmentation transforms, called during training and prediction and various heper fucntions
