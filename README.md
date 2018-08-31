# Audio Processing Project
Introduction and Information about the dataset
Our work mainly focuses on using CNN(Convolutional Neural Networks) for Acoustic Scene
Classification. For the project work we use the data set from Kaggle, the competition was organised
by TUT. The dataset consists of 15 different contexts such as beach, home etc and the data is made
up of mel-spectra on 10 seconds long segments. The baseline system or accuracy of system
predicting all the samples belong to one class is 0.06 or 6 p.c. accuracy.
Dataset provided by the organizers:
● X_train—a numpy array with dimensions 4500 * 40 * 501, which can be interpreted as
4500- training samples, spectrogram size- 40 – bin index and 501- time axis.
● X_test- numpy array with dimensions 1500 * 40 * 501 and the contents are similar to that of
X_train.
● y_train- which is a csv file with training labels and total number of categories or classes are
15. They are represented in order corresponding to the contents of training data set.
● Crossvalidation_train- which also contains additional data, which are split by default into
either test or train data set. In our code, we use the contents from this dataset and add it to
our training and testing datasets by which we can get additional data for training and
evaluation.

Proposed Model
Our idea is to implement the sound-net architecture and use Stochastic Gradient Descent as the
optimizer and see how the model performs. Since Sound-networks only on raw data form and it is
widely used to perform sound classification from unlabelled video, we decided to tweak the
architecture a bit. Moreover, the proposed idea was 1-D convolution we changed it into 2-D
convolution. There were two alternatives for Sound-Net, we can either use 5 layer network or 8 layer
network. We decided to use the 5-layer network since the amount of data available was not that
sufficient for such a deep network. This was the rationale behind choosing the architecture(to be
mentioned below).
The model or architecture is made of 5 convolution layers, 2 max Pooling layers, 1 flatten and 1
dense layers. There is a possibility to add a dropout layer to avoid overfitting but as now we are
using early stopping to avoid overfitting.

Trial-5
● Optimizer- SGD
● Loss Function- Categorical Cross Entropy
● No of Epochs- 250
● Data augmentation using Image Generators implemented
The variation of validation accuracy per epoch is bit high compared to SGD but there is an
considerable improvement accuracy. The best validation accuracy is 74.10 p.c.

Conclusion
So based on our analysis both SGD and Adam optimizers perform considerably better with the
advent of data augmentation. Further scope of improvement would be implementing a way to adjust
the learning rate, maybe it will help us achieve better results. There is further scope of improvement
which could be implemented to get better results.(In order to avoid over textual content I have just uploaded the best results obtained).
Reference:
Soundnet architecture-http://carlvondrick.com/soundnet.pdf
Datasets and other details can be found at :https://www.kaggle.com/c/acoustic-scene-2018 and the
kernel discussions helped us to begin our process.

This project was done as a part of Audio Processing Course along with Shapatrshi Basu.
