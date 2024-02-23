# nn-zero-to-hero
Building and delibrate practice of machine learning concepts. Following Andrej Karpathy's [Neural Network: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series

## Building micrograd

micrograd is a tiny neural net engine which implements back propagation. It construct a neural net, made up of layers of neurons. A neuron consist of individual mathematical operations which are constructioned into a DAG. The DAG is traversed for forward and backward pass to train the neural net for a training set. All this is done using python classes.

## Building makemore: bigrams and simple neural network

makemore takes names and trains a model on this dataset. Once trained it will make more names in the same style, but unique.

This series builds from using a bigram character model to more complex transformer models.

In this lecture a bigram character model is built. A bigram model is a pair of consecutive character. Where one letter is used to predict the second character using a lookup table. q ... u, for example!

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

In this part, data is loaded from a file into input and output lists as a training set. This contain values from 1 to 27 for each character and start/stop char '.'. An NN model which has 27 inputs and 27 outputs is created, effectively a 27,27 matrix with random negative and positive values. This model is to be trained. The input training set is one hot encoded for each item resulting in x,27 xenc shape. These are matmul together (x,27) giving logcounts. a softmax is used on these values to turn them into probabilities. exp(), then sum(1). The loss is calculated by working out the prob of the next letter, taking its log, summing it over the training set, averaging it and taking the negative. 

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





### Links
- [torch broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [torch multinominal](https://pytorch.org/docs/stable/generated/torch.multinomial.html)
- maximum likelihood estimation






## install 

Setup by using a virtual environment and running:

```
pip install jupyter torch numpy matplotlib
```

## other github attempts

https://github.com/Matjaz12/Neural-Networks-Zero-to-Hero/tree/main