# Tourbillon
 
## Dependencies
requirements.txt lists the requirements to run the repo. Run the command pip install -r requirements.txt in order to install all dependencies for this repo.

## Circular Autoencoders
Contains the main files required to train single layer autoencoders using the recirculation, backpropagation, and feedback alignment methodologies. Separated by the dataset used (Mnist, Fashion Mnist, and CIFAR-10)

### Each file takes the following arguments:

(*Required*) training_type: Training type for the model (Recirculation, RBP, BP)

batchsize: Input batch size for training (default: 64)

learning_rate: Learning rate for the classifier (default: 0.001)

device: Device used to run program (default: cpu)

seed: Seed used for reproducibility (default: 101)

ncirc: Number of CAE recirculations (default: 1)

