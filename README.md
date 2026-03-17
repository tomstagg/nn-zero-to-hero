# nn-zero-to-hero

Building skills in machine learning concepts by following Andrej Karpathy's [Neural Network: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series.

## Learning the fundamentals of neural networks

| Lesson | Title                                          | Description                                                                                                          | Status |
| ------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------ |
| 01     | micrograd                                      | Building a backpropagation engine and MLP from first principles, used for classification                             | ✅ Done |
| 02     | makemore pt1 — bigrams and neural network      | Building a bigram character model and basic neural network, trained on names to generate new ones                    | ✅ Done |
| 03     | makemore pt2 — MLP                             | Building an MLP with character embeddings following Bengio et al. 2003                                               | ✅ Done |
| 04     | makemore pt3 — activations, gradients, BatchNorm | Investigating MLP internals, Kaiming init, and batch normalisation                                                 | ✅ Done |
| 05     | makemore pt4 — becoming a backprop ninja       | Manual backpropagation through a neural net without using PyTorch autograd                                           | 🔜 Todo |
| 06     | makemore pt5 — WaveNet                         | Implementing a WaveNet-style hierarchical architecture for name generation                                           | 🔜 Todo |
| 07     | Let's build GPT                                | Building a Generatively Pretrained Transformer (nanoGPT) from scratch                                               | 🔜 Todo |
| 08     | Let's build the GPT Tokenizer                  | Building a BPE tokenizer from scratch, as used in GPT-2 and GPT-4                                                   | 🔜 Todo |
| 09     | Let's reproduce GPT-2                          | Reproducing the GPT-2 (124M) model and training it                                                                  | 🔜 Todo |

## Lesson 01: Building micrograd

Building a multi-layer perceptron, a neural net from first principles to classify datasets.

micrograd is a tiny neural net engine which implements backpropagation. The neural net consists of layers of neurons. A neuron is made of individual mathematical operations which are constructed into a DAG. A Value object which holds the graph of mathematical operations such as `+`, `-`, `*`, `/`, `pow()`, `exp()`,`tanh()` is developed and used to model a perceptron. Perceptrons are ordered into layers to form a multi-layer perceptron (MLP), the neural net. The Value object calculates the partial derivative to determine the gradient with respect to the final outputs for each Value.

The model is trained against a simple test training set and the loss calculated. Backpropagation through the Value object is used to determine the changes to the weights and bias of each perceptron neuron. The aim is to reduce the loss of the output over a number of training runs. The model is used to determine a hyperplane which is used to classify input vectors.

This lesson covers:

- setup jupyter to build, test and visualise neural networks
- walkthrough of partial differentials and backpropagation using graphviz
- building a Value object with the required math operations and associated partial differentials for backpropagation
- building activation functions
- using PyTorch as a comparison
- creating the nn library based on the Value object
- writing the loss function and training the network
- using the model to make predictions

## Lesson 02: Building makemore: bigrams and basic neural network approach

makemore takes names and trains a model on this dataset. Once trained it will make more names in the same style, but unique and hopefully name-like.

The lesson focuses on building a bigram character model to predict next letters to create new names. Given an input character, the logits output defines a character probability distribution from which the next character is sampled.
The second part of the lesson uses a neural network approach, with PyTorch, to achieve the same result.

A bigram model is a pair of consecutive characters. The first letter is used to predict the second character using a probability distribution. The model is built using 32K training words. The probabilities from this model are used for next letter prediction given a starting character.

This covers:

- parsing words into combinations to build a bigrams model, loading the frequencies into a tensor
- visualising the distribution using matplotlib
- using torch multinomial to sample from the distribution to extract the next letter
- investigating torch broadcast semantics and using broadcasting for summing rows to improve efficiency
- using the model to generate new names
- investigating what loss means for this data and how to calculate negative log likelihood
- adding model smoothing to prevent infinite loss

The second part of the lesson focuses on creating a basic neural net solution and training it on the words to generate new names.
The neural net is developed for both bigram and trigram cases. It simply consists of a tensor of weights associated to each input character which is used to define an output vector of logits, the unnormalised predictions. These are normalised using a softmax function to provide a distribution of probabilities for the next character. The negative log likelihood is calculated from these outputs and used to adjust the weights using gradients calculated via backpropagation. Concepts are explored in depth before using higher-level functions to simplify.

This covers:

- creating training sets for inputs and outputs
- creating a matrix of weights for the neural net
- using one hot encoding for the input training set
- normalising the nn output using softmax function - taking logits, converting to counts and producing a probability distribution
- calculating the loss, the negative log likelihood
- upgrading model with 2 x 27 one hot encoded inputs for a trigram nn model
- using the model with torch.multinomial to sample output characters to build new names
- splitting dataset into training, dev and test datasets and evaluating on them
- model smoothing as regularisation loss
- simplification without one_hot and use of cross_entropy instead

## Lesson 03: Building makemore: MLP

To improve the model we need to take more characters as inputs. We tried this as a bigram and a trigram model in the previous section. The problem with this is the model becomes exponentially complex with each additional input char, 27^3 = 19,600.

Instead, we'll follow the paper [A Neural Probabilistic Language Model - Bengio et al.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

This model calculates an embedding for each input character. An embedding is a character feature vector, from 2 to 30 dimensions in our case. It generalises the character features. This x character input is converted to x embeddings which are manipulated with pytorch to provide a concatenated embedding for a hidden layer. The hidden layer of neurons is a fully connected layer of 100 - 300 neurons which are connected to 27 output neurons, providing the logits vector which is used to calculate, via a softmax function, the next character probability distribution. Various hyperparameter configuration options and simplifications are made with the model trained on a subset of the full dataset, and evaluated on dev/validation and test sets. Backpropagation is used to adjust the model weights to minimise the loss.

The lesson covers:

- setup of initial char lookup embedding
- using a character block variable for generation of a dataset to train the model, initially 3 char inputs
- investigating options for encoding the input dataset with embedding - one hot vs tensor index
- manipulation of the embedding to allow multiplication with the hidden layer - using pytorch internals, and seeing that data is contiguously stored
- using cross entropy rather than explicit calculations to deal with logits that are extreme
- using mini-batches to speed up each iteration
- working out the learning rate to use
- using training, dev and test sets to avoid overfitting
- viewing loss progress over iterations
- visualising 2d character embedding
- trialing different hyperparameters to determine loss in training sets
- sampling from the model

Exercises involve attempting to find the best hyperparameter configuration to minimise the loss. Checking loss against the dev and test datasets to avoid overfitting the data for the training set. Also, tuning the initial model weights to get a better starting loss, by making weights uniform.

## Lesson 04: Building makemore: Activations, Gradients and BatchNorm

Focusing on the internals of MLPs and investigating the initial state and fixing it for the initial loss and saturated tanh activation function. Rather than setting these manually we calculate the initial weights using a "Kaiming init" approach. We investigate batch normalisation to control the statistics of activations in the neural net.

We calculate the initial scaling using "Kaiming init" from the [Kaiming init paper](https://arxiv.org/abs/1502.01852). This is defined in pytorch at [torch.nn.init.kaiming*normal*](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) where gains are defined for different activation functions.

Still focused on MLP — why it's hard to optimise, by looking at gradients and activations and how they behave during training.

#### Fixing the initial loss

The initial loss is based on a random distribution of weights. Loss is ~27.9, whereas a uniform distribution would be close to 3.3.
Weights are initialised with extreme values which mean that many of the training runs initially are spent reducing this to more normal values.
This gives the hockey stick appearance for loss over runs. What we can do is set the bias to 0 and the W to a fraction of their normalised
values to give a better starting loss. We could set the weights also to 0 but for some reason it's better if they are not zero.

#### Fixing the saturated tanh

tanh at extreme values takes on values of either 1 or -1. At these points the gradient is very close to 0 and so movement away from these values
is extremely hard. If the hidden layer after passing through the tanh function is 1 or -1 that is bad for each neuron. The optimisation is stuck
and the tanh function squashes the input to values between 1 and -1. It's possible to see for each training run if the tanh is in this bad position.
If this occurs for all training examples the neuron is dead — permanent brain damage has occurred. It can also occur during
training if the learning rate is too high; this can knock out neurons into the extreme value region and they never come back as the gradient is 0.

The way to prevent this is to scale the weights and bias feeding into tanh, the preactivations, which normalises the input. This makes
the distribution out of tanh more uniform and less likely to be > 0.99.

## Kaiming Normal

When we multiply our randomised inputs with randomised weights of the initial layer, we increase the standard deviation. We want to normalise our weights by the scale of the fan-in. We can use [Kaiming Normal](https://pytorch.org/docs/stable/nn.init.html) to set the gain correctly.

Accurate initialisation used to be critical as the network was fragile. It is now less important as there are various techniques like optimisers, residual connections, and residual layers.

## Batch Normalisation

Batch normalisation is used to control the statistics of activations in the neural net. BatchNorm layers are added to the network after matmul, linear or convolutional layers to ensure the activations are not too high or low. Without them, the non-linear activation function (tanh) can become saturated, going to +/- 1 with a gradient of 0. The BatchNorm layer has gain and bias parameters trained using backprop, and two buffers — mean and std — which are updated via running statistics rather than backprop. The buffers are used to centre the input batch to a unit Gaussian, then scaled with the gain and shifted using the bias parameters.

Brief overview of ResNet used for CNN. Implementation [here](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py). In the `Bottleneck` class you can see `conv1` - the linear layer, `bn1` the BatchNorm and `relu` the activation (non-linear) layer. The `__init__` is the model initialisation. The `conv1x1` and `conv3x3` are initialised with bias = False, in the same way as we have removed the bias from the linear layer.

Focusing on PyTorch equivalents of our code:

- **nn.Linear** - Wx + b. Need to know the fan in and fan out, including an optional bias. You would not need a bias if the layer is followed by a normalisation layer. Weights are initialised similar to Kaiming init but without the gain.
- **nn.BatchNorm** - input is the hidden layer size (200 for us), and the eps which is for numeric stability, and momentum — which should be high or low depending on batch size. You need to avoid thrashing around, so momentum should be low if the batch size is small, and can be higher for more representative batches. `track_running_stats = True` is used if you are calculating batch mean and std during the training run, as opposed to separately as a follow-on step after training.

# Summary

The importance of setting activations and gradients and their statistics in a neural net becomes critical as networks get deeper. If the activations are too confident initially, you get hockey stick loss. Avoiding this gives better initial and final loss. We discussed setting weights closer to zero initially. This approach doesn't scale to deeper networks with many layers — it is much harder to set weights to be uniform across the whole network. This is where normalisation comes in; BatchNorm is one implementation of this. You can put it throughout the network, and if you want unit Gaussian activations you can enforce that (mean=0, std=1). We folded the running mean and std into the training loop, updated via a momentum term. No one likes the BatchNorm layer, as it couples training examples; you can instead use LayerNorm or RMSNorm in more recent deep learning.

### Links

- [torch broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [torch multinominal](https://pytorch.org/docs/stable/generated/torch.multinomial.html)
- maximum likelihood estimation

## Installation

Using `uv` to manage project and dependencies.
Install graphviz using homebrew : `brew install graphviz`

running jupyter: `uv run --with jupyter jupyter lab`

troubleshoot jupyter venv: use "Python 3 (ipykernel)" in top right

### Notebook git filter (nbstripout)

`nbstripout` is installed as a dev dependency and strips cell outputs from notebooks before staging, keeping diffs clean.

After cloning, install the git filter manually:

```bash
uv run nbstripout --install
```

Then update the filter commands in `.git/config` to use `uv run` rather than the hardcoded venv path:

```bash
git config filter.nbstripout.clean "uv run nbstripout"
git config filter.nbstripout.smudge cat
git config diff.ipynb.textconv "uv run nbstripout -t"
```

The filter runs automatically on `git add` — no manual steps needed day-to-day. Your local notebook outputs are preserved in the editor; only the stripped version is committed.

## other github attempts

https://github.com/Matjaz12/Neural-Networks-Zero-to-Hero/tree/main
