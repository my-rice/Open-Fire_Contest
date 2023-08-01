In train.ipynb we have organised the code in such a way that at the beginning we have sections of code common to all trainings related to downloading and decompressing videos, a section to extract frames at a specific frame rate from the videos, a section to create the dataset from the frames. Other common code sections are functions to start and configure the Tensorboard and functions to perform K-fold cross-validation. 

Then, we have the code sections specific to each training. In each section we have code to create the model, code to configure the training parameters, code to train the model and code to save the training data.

We trained several models, each in a different code section (we named these sections with "Attempt x: ModelName", where we specify which neural network architecture was used for the training). The proposed models are as follows:
 - Attempt 1: MobileNetV2
 - Attempt 2: MobileNetV3Small
 - Attempt 3: ResNet50
 - Attempt 4: FireNetV2
 - Attempt 5: ResNet18

In order to test the models produced by the training, we created a test.py script which, after choosing the neural network model to be used, classifies the videos as FIRE or NOFIRE and places the results in the specified folder on the command line.

In order to obtain metrics characterising the performance of a model, we created the metrics.py script, which interprets the results produced by test.py and produces useful information (accuracy, precision, recall, etc.) to assess the network's performance.

The model.py file facilitates the configuration of the neural network, allowing a pre-trained model to be instantiated where needed.

The files firenet.py and firenetV2.py implement two different neural network architectures.

Finally, converter.py is a file that enables the conversion of videos in .avi format to .mp4 format (used to convert videos for the test set).
