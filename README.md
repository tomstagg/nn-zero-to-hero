# nn-zero-to-hero

Building skills in machine learning concepts by following Andrej Karpathy's [Neural Network: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series.

## Lesson 01: Building micrograd

Building a multi-layer perception, a neural net from first principles to classify datasets.

micrograd is a tiny neural net engine which implements back propagation. The neural net consists of layers of neurons. A neuron is made of individual mathematical operations which are constructed into a DAG. A Value object which holds the graph of mathematical operations such as `+`, `-`, `*`, `/`, `pow()`, `exp()`,`tanh()` is developed and used to model a perceptron. Perceptrons are ordered into layers to build a multi-layer perceptron (MLP), the neural net. The Value object calculates the partial derivation to determine the gradient with respect to the final outputs for each Value.

The model is trained against a simple test training set and the loss calculated. Back propagation through the Value object is used to determine the changes to the weights and bias of each perceptron neuron. The aim being to reduce the loss of the output over a number of training runs. The model is used to determine a hyperplane which is used to classify input vectors. 

This lesson covers: 
- setup jupyter to build, test and visualise neural networks
- walkthrough of partial differentials and backprogration using graphviz
- building a Value object with required math operations required with associated partial differentials for back propagation
- building activation functions
- using PyTorch as a comparison
- creating the nn library based on the Value object
- writing the loss function and training the network
- using the model to make predictions

## Lesson 02: Building makemore: bigrams and basic neural network approach

makemore takes names and trains a model on this dataset. Once trained it will make more names in the same style, but unique and hopefully name-like.

The lesson focuses on building a bigram character model to predict next letters to create new names. Given an input character, the logits output defines a character probability distribution from which the next character is sample. 
The second part of the lesson use a neural network approach, with PyTorch, to achieve the same result.

A bigram model is a pair of consecutive characters. The first letter is used to predict the second character using a probability distribution. The model is built using 32K training words. The probabilities from this model are used for next letter prediction given a starting character. 

This covers: 
- parsing words into combination to build a bigrams model, loading the frequencies into a tensor
- visualising the distribution using matplotlib
- using torch multinomial to sample from the distribution to extract the next letter
- investigating torch broadcast semantics and using broadcasting for summing rows to improve efficiency
- using the model to generate new names
- investigating what loss means for this data and how to calculate negative log likelihood
- adding model smoothing to prevent infinite loss

The second part of the lesson focuses on creating a basic neural net solution and train it on the words to generate new names.
The neural net is developed for both bigram and trigram cases. It simply consists of a tensor of weights associated to each input character which is used to define an output vector of logits, the unnormalised predictions. These are normalised using a softmax function to provide a distribution of probabilities for the next character. The negative loss likelihood is calculated from these outputs and used to adjust the weights using gradients calculated via backpropogation. Concepts are explored in depth before using higher-level functions to simplify. 

This covers:
- creating training sets for inputs and outputs
- creating a matrix of weights for the neural net
- using one hot encoding for input training set
- normalising the nn output using softmax function - taking logits, convert to counts and producing a probability distribution
- calculating the loss, the negative log likelihood
- upgrading model with 2 x 27 one hot encoded inputs for a trigram nn model
- using the model with torch. multinominal to sample output characters to build new names
- splitting dataset into training, dev and test datasets and evaluating on them
- model smoothing as regularisation loss 
- simplification without one_hot and use of cross_entropy instead

## Lesson 03: Building makemore: MLP

To improve the model we need to take more characters as inputs. We tried this as a bigram and a trigram model in the previous section. The problem with this is the model becomes exponentially complex with each additional input char,  27^3 = 19,600. 

Instead, we'll follow the paper [A Neural Probablistic Language Model - Bengio et al.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

This model calculates an embedding for each input character. An embedding is a character feature vector, from 2 to 30 dimension in our case. It generalises the character features. This x character input is convert to x embeddings which are manipulated with pytorch to provide a concatenated embedding for a hidden layer. The hidden layer of neurons is a fully connected layer of 100 - 300 neurons which are connected to 27 output neurons, providing the logits vector which is used to calculate via a softmax function, the next character probability distribution. Various hyperparameter configuration options and simplifications are made with the model trained on subset of the full dataset, and evaluated on dev/validation and test sets. Back propagation is used to adjust the model weights to minimise the loss. 

The lesson covers:
- setup of initial char lookup embedding
- using a character block variable to generation of dataset to train the model, initially 3 char inputs
- investigate options to encoding input dataset with embedding - one hot vs tensor index
- manipulation of embedding to allow multiplication with hidden layer - using pytorch internals, and seeing that data is contiguously stored
- using cross entropy rather than explicit calculations to deal with logits that are extreme
- using mini-batches to speed up each iteration
- working out the learning rate to use
- using training, dev and tests set to avoid overfitting
- viewing loss progress over iterations
- visualising 2d character embedding
- trialing different hyperparameters to determine loss in training sets
- sampling from the model

Exercises involve attempting to find the best hyperparameter configuration to minimise the loss. Checking loss against the dev and test datasets to avoid overfitting the data for the training set. Also, tuning the initial model weights to get a better starting loss, by making weights uniform.

## Lesson 04: Building makemore: Activations, Gradients and BatchNorm

Focussing on the internals of MLPs and investing the initial state and fixing if for the initial loss and saturated activation tanh function. Rather than setting  these manually we calculate the initiation weights using a "kaiming init" approach. We investigate batch normalisation to control the statistics of activiations in the neural net.




We calculate the initialse scaling using "Kaiming init" from the [Kaiming init paper](https://arxiv.org/abs/1502.01852). This is defined in pytorch at [torch.nn.init.kaiming_normal_](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) where gains are defined for different activation functions. 

Still focussed on MLP. why it's hard to optimise, by looking at gradients and activitiation and how they behave during training. It's harder to optimise these.

#### Fixing the initial loss
The initial loss is based on a random distribution of weights. loss is ~27.9, whereas a uniform distribution would be close to 3.3. 
Weights are initalised with extreme values which mean that many of the training runs initially are to reduce this to more normal values.
This gives the hocky stick appearance for loss for runs. What we can do is set the bias to 0 and the W to a fraction of there normalised
values to give a better starting loss. We could set the weights also to 0 but for some reasons, it's better if they are not zero. 

#### Fixing the saturated tanh

tanh at extreme values takes on values of either 1 or -1. at these points the gradient is very close to 0 and so movement away from these values
is extremely hard. If the hidden layer after passing through the tanh function is 1 or -1 that that is bad for each neuron. The optimisation is stuck
and. the tanh function squash the input to values between 1 and -1. It's possible to see for each training run if the tanh is in this bad position. 
If this occurs for all training examples this is bad and the neuron is dead! Permanent brain damage has occured. It can also occur during 
training of the learning rate is too high, this can knock out neurons into the extreme value region and they never come back as the grad is 0.

The way to prevent this is to set scale the weights and bias feeding into tanh, the preactivations, which normalises the input. This makes
the distribution out of tanh more uniform and less likely to be > 0.99. 

## Kaiming Normal
When we multiply our randomised inputs with randomised weights of the initial layer, we increase standard deviation. We want to normalise our weights by the scale of the fan-in. We can use [kaiming normal](https://pytorch.org/docs/stable/nn.init.html) to set the gain correctly. 

It used to be important to set the initialisation accurately as the network was fragile. Now less important as there are various techniques like optimises, residual connection, residual layers.

## Batch Normalisation

Batch normalisation is used to control the statistics of activiations in the neural net. batchNorm layers are added to the network, after matmul, linear or convulational layers to ensure the activations are not too high and low. If they weren't there then the non-linear activations function, tanh, can become saturated, going to +/- 1 and the gradient is 0. The batchnorm has gain and bias parameters, trained using backprop and two buffers, mean and std which are not trained using backprop. The buffers are used to center input batch to unit gaussian and then scaling with gain and shifting using the bias parameters.

Brief overview of Resnet used for CNN. Implementation (here)[https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py]. In the `Bottleneck` class you can see `conv1` - the linear layer, `bn1` the batchNorm and `relu` the activation (non-linear) layer. The `__init__` is the model initalisation. The `conv1x1` and `con3x3` are initalised with bias = False, in the same way as we have removed the bias from the linear layer.


Thashing, momentum high or low dependent on batch size. Looking at nn.Linear and nn.batchnorm and walking through the parameters.
avoiding hockey stick loss, controlling activations squashed to zero or explope to inf, need roughly gaussian activations using batchnorm. 
Difficult to do with many layers to have roughly uniform activations acorss the network, that's when batchnorm helps, one implementation of normalisation which you can sprinkle about the network. It's differentialable. 


### Links
- [torch broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [torch multinominal](https://pytorch.org/docs/stable/generated/torch.multinomial.html)
- maximum likelihood estimation


## Installation 

Using `uv` to manage project and dependencies.
Install graphviz using homebrew : `brew install graphviz`

running jupyter: `uv run --with jupyter jupyter lab`

troubleshoot jupyter venv: use "Python 3 (ipykernel)" in top right


## other github attempts

https://github.com/Matjaz12/Neural-Networks-Zero-to-Hero/tree/main