# nn-zero-to-hero

Building skills in machine learning concepts by following Andrej Karpathy's [Neural Network: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series.



## Lesson 01: Building micrograd

Building a multi-layer perception, a neural net from first principles to classify datasets.

micrograd is a tiny neural net engine which implements back propagation. The neural net consists of layers of neurons. A neuron is made of individual mathematical operations which are constructioned into a DAG. A Value object which holds the graph of mathematical operations such as `+`, `-`, `*`, `/`, `pow()`, `exp()`,`tanh()` is developed and used to model a perceptron. Perceptrons are ordered into layers to build a multi-layer perceptron (MLP), the neural net. The Value object calculates the partial derivation to determine the gradient with respect to the final outputs for each Value.

The model is trained against a simple test training set and the loss calculated. Back propogation through the Value object is used to determine the changes to the the weights and bias of each perceptron neuron. The aim being to reduce the loss of the output over a number of training runs. The model is used to determine a hyperplane which is used to classify input vectors. 

This lesson covers: 
- setup of jupyter to build, test and visualise neural networks
- walkthough of partial differentials and backprogration using graphviz
- building a Value object with required math operations required with associated partial differentials for back propogation
- building activation functions
- using pytorch as a comparision
- creating the nn library based on the Value object
- writing the loss function and training the network
- using the model to make predictions

## Lesson 02: Building makemore: bigrams

makemore takes names and trains a model on this dataset. Once trained it will make more names in the same style, but unique.

The lesson focuses on building a bigram character model to predict next letters to create new names. The second part of the lesson use a neural network approach, with pytorch, to achive the same result. The model is traind

A bigram model is a pair of consecutive characters. The first letter is used to predict the second character using a probability distribution.

A bigram model is built using 32K training words. The probabilities from this model are used for next letter prediction given a starting character. 

The following are explored:
- parsing words into combination to build a bigrams model, loading the frequencies into a tensor
- visualising the distribution using matplotlib
- using torch multinomial to sample from the distribution to extract the next letter
- investigating torch broadcast semantics and dangers, and using broadcasting for suming rows to improve efficiency
- using the model to generate new names
- investigating what loss means for this data and how to calculate negative log likelihood

The second part of the lesson focusses on creating a neural net solution and train it on the words to generate new names.

This covers:
- creating training sets for inputs and outputs
- creating a matrix of weights for the NN
- using one hot encoding for input training set
- normalising the nn output using softmax function - taking logits, convert to counts and producing a prob distribution
- calculating the loss, the negative log likelihood



## Building makemore: bigrams and simple neural network


This series builds from using a bigram character model to more complex transformer models.

In this lecture a bigram character model is built. A bigram model is a pair of consecutive character. Where one letter is used to predict the second character using a lookup table. q -> u, u is predicts to follow q for example.

The bigram character model loads words into a list, creates bigrams from these words, stores the results in a model - a tensor array, creates probabilities from this model for next letter prediction given a starting character and generates new words. 

These concepts are explored:
- loading data into a list from a file
- using zip to create bigram characters
- visually displaying the model using matplotlib
- tensor broadcasting semantics for summed tensors
- using torch multinominal to predict next character
- calculate the loss for the model
- model smoothing

The makemore neural net uses the same data to create a training set, initalises weights in a model, and calculate the loss of the model and iteratively tune the weights using forward and backward passes. A softmax function is defined to convert model weights to a probability distribution which is used to generate outputs.

In this part, data is loaded from a file into input and output lists as a training set. This contain values from 1 to 27 for each character and start/stop char `.`. An NN model which has 27 inputs and 27 outputs is created, effectively a `(27,27)` matrix with random negative and positive values. This model is to be trained. The input training set is one hot encoded for each item resulting in x,27 xenc shape. These are matrix multiplied (matmul) together `(x,27)` giving logcounts. a softmax is used on these values to turn them into probabilities. exp(), then sum(1). The loss is calculated by working out the prob of the next letter, taking its log, summing it over the training set, averaging it and taking the negative. 

## Lesson 02: 

## building makemore: MLP

To improve the model we need to take more characters as inputs. We tried this as a bigram and a trigram model in the previous section. The problem with this is the model blows up exponentially 27^3 = 19,600. 

Instead we'll follow the paper [A Neural Probablistic Language Model - Bengio et al.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

This model takes multiple previous characters and looks up an embedding, a character feature vector, from a matrix. The embedding is used to generalise. Rather than having 27 options, we have the embedding for each character, in our case a 2 dimensional representation.

A hidden layer which can be any size, but is connected to each character's feature vector. Finally an output softmaxed layer is fully connected to the hidden layer and outputs a probability distribution for next character.

All is optimised via back propogation.

Hyperparameters are used to change the embedding dimensions and hidden layer size, as well as training parameters.
This associates feature embeddings with a characters to generalise knowledge and takes multiple previous characters. 

Topics covers are: 
- setup initial char lookup embedding
- using a character block variable to generation of dataset to train the model, initially 3 char inputs
- investigate options to encoding input dataset with embedding - one hot vs tensor index
- manipulation of embedding to allow multiplication with hidden layer - using pytorch internals as data is continguously stored
- using cross entropy rather than explict calcs to deal with logits that are extreme
- using mini-batches to speed up each iteration
- working out the learning rate to use 
- using training, dev and tests set to avoid overfitting
- viewing loss progress over iterations
- visualising 2d character embedding
- trialing different hyperparameters to determine loss in training sets
- sampling from the model

## building makemore: activations and gradient, batchnorm

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

### Links
- [torch broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [torch multinominal](https://pytorch.org/docs/stable/generated/torch.multinomial.html)
- maximum likelihood estimation


## install 

Using `uv` to manage project and dependencies.
Install graphviz using homebrew : `brew install graphviz`

running jupyter: `uv run --with jupyter jupyter lab`


## other github attempts

https://github.com/Matjaz12/Neural-Networks-Zero-to-Hero/tree/main