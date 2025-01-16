$\newcommand{\bigO}{O}
\newcommand{\trsp}[1]{#1^\intercal} % transpose
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\DeclareMathOperator*{\vari}{\mathbb{V}}   % Probability
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\sigm}{\phi_{\text{sig}}} % logistic function
\newcommand{\bigOmega}{\Omega}
\newcommand{\softmax}{\textsf{soft}}
\newcommand{\KL}{\textrm{D}_\textrm{KL}} % Kullback-Leibler divergence
\newcommand{\twovec}[2]{\begin{pmatrix}#1\\#2\end{pmatrix}}
\newcommand{\rank}{\text{rank}}
\newcommand{\diag}{\text{diag}} % diagonal matrix
\newcommand{\ph}[1]{\mathsf{#1}} % general polyhedron
$

Neural networks 
===============

Let's train a neural network.[^Niel]
Normally, SGD or one of its more advanced cousins is used
to train neural networks. The main issue is how to compute the gradient as efficiently as 
possible. But let's not get ahead of ourselves. 

[^Niel]: A very good source for this -- and other aspects of neural networks! -- is Michael Nielsen's [website](http://neuralnetworksanddeeplearning.com)

The neural network we consider has $K$ layers, each layer $k$ consisting of $n_k$ nodes.
The last layer may either consist of a single node (for binary classification) or 
of several nodes. Normally, the activation function of any hidden layer 
is ReLU ($x\mapsto \max(0,x)$) or leaky ReLU  $(x\mapsto \max(\alpha x,x)$ for $\alpha\in (0,1)$),
while the activation layer for the output layer is either the logistic function 
(for a single output node) or softmax (for several output nodes). We will simply
write $\sigma_k$ for the activation function of layer $k$, and I will pretend 
that $\sigma_k$ is differentiable -- even though this is clearly a lie. Let me discuss
this issue later.

```{figure} pix/net.png
:name: netfig
:width: 15cm
A classfication ReLU-neural network
```

Set $f^{(0)}(x)=x$.
For the layers $k=1,\ldots, K$ we compute the input of the $k$th layer as

$$
g^{(k)}(x)=W^{(k)}f^{(k-1)}(x)+b^{(k)}
$$

and for $k=1,\ldots, L-1$ the output as 

$$
f^{(k)}(x)=\sigma_k(g^{(k)}(x)),
$$

where $\sigma_k$ is applied componentwise.
The output of the network is then $f^{(K)}(x)$.
For simplicity, let's assume a single output node, so that we have $f^{(K)}_1(x)$
as output. Also, it's safe to assume that the activation function of the output is logistic.

(backpropsec)=
Back propagation
----------------

To train the neural network, we need to specify a loss function. Historically this was often the 
*square loss*. That is, given a training set $S\subseteq \mathcal X\times\mathcal Y$
the empirical risk was taken to be

$$
\frac{1}{|S|}\sum_{(x,y)\in S}||f^{(K)}(x)-y||^2
$$

In modern networks, square loss has been replaced by other loss functions -- but we will 
discuss that later. In any case, we assume that we have fixed some 
loss function that we will see as depending on the weights of the neural network, and that is 
the average of losses at specific data points in the training set:

```{math}
:label: netloss
L(a)=\frac{1}{|S|}\sum_{(x,y)\in S}L_{(x,y)}(a)
```

For SGD we now have to compute $\nabla L_{(x,y)}$ given some randomly drawn $(x,y)\in S$.
How can this be done as efficiently as possible? With the *back propagation* algorithm!

So, let's fix some datapoint $(x,y)\in S$. To simplify notation and because we are now interested 
in how the output of the network changes as a function of the *weights* and not 
as a function of the input, we drop the reference to the input $x$ from the intermediate variables
$f^{(k)}(x)$ and $g^{(k)}(x)$. That is, we simply write  $f^{(k)}$ and $g^{(k)}$.

For \alg{SGD}, the quantity we have to compute is $\nabla_a L_{(x,y)}$, where the subscript $a$
is meant to indicate that we take the gradient with respect to the weights $a$. In particular,
if $w^{(k)}_{hi}$ is the weight between the $h$th node of $k$th layer and the $i$th node
of the $k-1$th layer and if $b^{(k)}_h$ is the bias of the $h$th node of the $k$th layer,
then we need to compute the quantities

$$
\frac{\partial L_{(x,y)}}{\partial w^{(k)}_{hi}}\text{ and }
\frac{\partial L_{(x,y)}}{\partial b^{(k)}_{h}}
$$


Let's start with the bias $b^{(K)}_1$ of the output node and compute 

$$
\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{1}}.
$$

To do so we observe that $L_{(x,y)}$ is a function of $f^{(K)}_1$, while $f^{(K)}_1$ depends on $g^{(K)}_1$,
which in turn is a function of $b^{(K)}_1$:

$$
 f^{(K)}_1=\sigma_K(g^{(K)}_1) \text{ and }g^{(K)}=W^{(K)} f^{(K-1)}+b^{(K)}_1
$$

Using the chain rule of differentiation we get

$$
\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{1}} = \frac{\partial L_{(x,y)}}{\partial f^{(K)}_{1}}
\cdot  \frac{\partial f^{(K)}_{1}}{\partial g^{(K)}_1}
\cdot  \frac{\partial g^{(K)}_1}{\partial b^{(K)}_1}
$$

In the same way, we get for the weights between the penultimate and the output layer

$$
\frac{\partial L_{(x,y)}}{\partial w^{(K)}_{1i}} = \frac{\partial L_{(x,y)}}{\partial f^{(K)}_{1}}
\cdot  \frac{\partial f^{(K)}_{1}}{\partial g^{(K)}_1}
\cdot  \frac{\partial g^{(K)}_1}{\partial w^{(K)}_{1i}}
$$

The two expressions contain a common factor that we call $\delta^{(K)}_1$:  

```{math}
:label: deltaK
\delta^{(K)}_1:=\frac{\partial L_{(x,y)}}{\partial g^{(K)}_1} = 
\frac{\partial L_{(x,y)}}{\partial f^{(K)}_{1}}
\cdot  \frac{\partial f^{(K)}_{1}}{\partial g^{(K)}_1}
= \frac{\partial L_{(x,y)}}{\partial f^{(K)}_{1}} \cdot \sigma_K'(g^{(K)}_1)
```

We also calculate

$$
\frac{\partial g^{(K)}_1}{\partial b^{(K)}_1} = 1
\text{ and }
\frac{\partial g^{(K)}_1}{\partial w^{(K)}_{1i}} = f^{(K-1)}_{i}
$$

Thus 

$$
\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{1}} =
\delta^{(K)}_1
\text{ and }
\frac{\partial L_{(x,y)}}{\partial w^{(K)}_{1i}}= \delta^{(K)}_1\cdot f^{(K-1)}_{i}
$$


As an illustration, we compute the relevant values for the square loss (and logistic activation at the output),
ie, for $L_{(x,y)}(f^{(K)}_1)=(f^{(K)}_1-y)^2$ and $\sigma_K=\sigm$:

$$
\delta^{(K)}_1=  \frac{\partial L_{(x,y)}}{\partial f^{(K)}_{1}} \cdot \sigma_K'(g^{(K)}_1)
= 2( f^{(K)}_{1} -y ) \cdot \sigm'(g^{(K)}_1)
$$

Recall that 

$$
\sigm(z)=\frac{1}{1+e^{-z}}
$$

With a little bit of manipulation, we may see that $\sigm'(z)=\sigm(z)(1-\sigm(z))$ and thus

```{math}
:label: sqbw
\frac{\partial L_{(x,y)}}{\partial w^{(K)}_{1i}}
 & = 2( f^{(K)}_{1} -y )\cdot f^{(K)}_1(1-f^{(K)}_1) \cdot f^{(K-1)}_{i} \\
\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{1}} & = 2( f^{(K)}_{1} -y )\cdot f^{(K)}_1(1-f^{(K)}_1) 
```

Note that $\sigm(g^{(K)}_1)=f^{(K)}_1$.

Now, how do we calculate the gradient by weights in earlier layers? Motivated by {eq}`deltaK`
let's write

```{math}
:label: delta
\delta^{(k)}=\nabla_{g^{(k)}}L_{(x,y)} = \trsp{\left(
\frac{\partial L_{(x,y)}}{\partial g^{(k)}_1},\ldots,\frac{\partial L_{(x,y)}}{\partial g^{(k)}_{n_k}}
\right)}
```

Why is that useful? Because we can express the gradient by weights in terms of $\delta^{(k)}$:
```{math}
:label: gradb
\frac{\partial L_{(x,y)}}{\partial b^{(k)}_{h}} = 
\frac{\partial L_{(x,y)}}{\partial g^{(k)}_{h}}\cdot 
\frac{\partial g^{(k)}_{h}}{\partial b^{(k)}_{h}}
= \delta^{(k)}_h
```

as $g^{(k)}=W^{(k)}f^{(k-1)}+b^{(k)}$. In the same way we get
```{math}
:label: gradw
\frac{\partial L_{(x,y)}}{\partial w^{(k)}_{hi}} = 
\frac{\partial L_{(x,y)}}{\partial g^{(k)}_{h}}\cdot 
\frac{\partial g^{(k)}_{h}}{\partial w^{(k)}_{hi}}
= \delta^{(k)}_h\cdot f^{(k-1)}_i
```

How can we compute $\delta^{(k)}$ efficiently?
\begin{align*}
\delta^{(k)}_h & = \frac{\partial L_{(x,y)}}{\partial g^{(k)}_h}
= \sum_{i=1}^{n_{k+1}} \frac{\partial L_{(x,y)}}{\partial g^{(k+1)}_i}\cdot 
\frac{\partial g^{(k+1)}_i}{\partial g^{(k)}_h}\\
& = \sum_{i=1}^{n_{k+1}} \delta^{(k+1)}_i\cdot \frac{\partial g^{(k+1)}_i}{\partial g^{(k)}_h}
\end{align*}
Now, we observe that[^matnot]
$g^{(k+1)}_i=W^{(k+1)}_{i,\bullet}f^{(k)}+b^{(k+1)}_i=W^{(k+1)}_{i,\bullet}\sigma_{k}(g^{(k)})+b^{(k+1)}_i$.
It follows that

[^matnot]: for a matrix $A$, the notation $A_{i,\bullet}$ denotes the $i$th row of $A$.

$$
\frac{\partial g^{(k+1)}_i}{\partial g^{(k)}_h} = w^{(k+1)}_{ih}\cdot \sigma'_k(g^{(k)}_h)
$$

Thus
```{math}
:label: updatedelta
\delta^{(k)}_h = \sum_{i=1}^{n_{k+1}} w^{(k+1)}_{ih}\cdot \delta^{(k+1)}_i\cdot \sigma'_k(g^{(k)}_h)
```
With the help of the *Hadamard product* we can write this in a more compact way. 
The Hadamard product of two matrices (or vectors) $A,B$ of the same size 
gives the matrix obtained by element-wise multiplication. That is, 
the product is the matrix of same size defined by

$$
(A\odot B)_{ij}=A_{ij}\cdot B_{ij}
$$

Then
\begin{equation}
\delta^{(k)}=\trsp{(W^{(k+1)})}\delta^{(k+1)}\odot \sigma'_k(g^{(k)})
\end{equation}

Using {eq}`deltaK` and {eq}`gradb`--{eq}`updatedelta` we can formulate 
the *back propagation* algorithm. For greater generality, let us allow any number $n_K$
of output nodes.

```{prf:Algorithm} back propagation
:label: backprop

**Instance** A neural network, a loss function $L_{(x,y)}$ for a single datapoint $(x,y)$\
**Output** The gradient of $L_{(x,y)}$ by weights

1. Compute $\delta^{(K)}=\nabla_{f^{(K)}} L_{(x,y)}\odot \sigma_K'(g^{(K)})$
2. Set $\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{h}} = \delta^{(K)}_h$ for $h=1,\ldots, n_K$
3. Set $\frac{\partial L_{(x,y)}}{\partial w^{(K)}_{hi}} =  \delta^{(K)}_h\cdot f^{(K-1)}_i$ for $h=1,\ldots, n_K$, $i=1,\ldots,n_{K-1}$
4. **for** $k=K-1,\ldots, 1$:
5.   {{tab}}Compute $\delta^{(k)}=\trsp{(W^{(k+1)})}\delta^{(k+1)}\odot \sigma'_k(g^{(k)})$
6.   {{tab}}Set $\frac{\partial L_{(x,y)}}{\partial b^{(k)}_{h}} = \delta^{(k)}_h$ for $h=1,\ldots, n_k$
7.   {{tab}}Set $\frac{\partial L_{(x,y)}}{\partial w^{(k)}_{hi}} =  \delta^{(k)}_h\cdot f^{(k-1)}_i$ for $h=1,\ldots, n_k$, $i=1,\ldots,n_{k-1}$
8. **output** all $\frac{\partial L_{(x,y)}}{\partial w^{(k)}_{hi}}$ and $\frac{\partial L_{(x,y)}}{\partial b^{(k)}_{h}}$
```
We note that the algorithm is very efficient: for every layer (from last to first) the algorithm 
basically reduces to one matrix-vector multiplication (in Line 5). 

Training a neural network can often be sped-up if it's done on a GPU (graphics processor unit)
rather than on a CPU. GPU are highly specialised for graphics operations such as 
translations and rotations. In particular, they can perform matrix-vector operations very
efficiently and, to some extent, in parallel. I believe this means that 
the gradient of a minibatch can be computed in parallel on a GPU -- as long as the 
minibatch size is not too large. (I did not manage to find a credible source, though.)


How is a neural network trained? With SGD (or one of its more advanced cousins) 
together with back propagation to compute the gradients. 
This is not the whole story, though -- there are a lot of other aspects to consider. We will
discuss some.

This first easy issue concerns the most popular type of hidden units, the ReLU.
It is not differentiable at 0 and therefore, at least formally, computing the gradient
via back propagation makes no sense. However, that a the gradient of a ReLU
needs to be evaluated at 0 is quite unlikely, and returning back the left-derivative, 
namely 0, will work fine. (The right-derivative would be equally good.) 

Note, moreover, that in line 5 the derivative of the activation function needs to be computed.
For ReLU this is particularly simple:

$$
\text{ReLU}'(z)=\begin{cases}
1 & z>0\\
0 & z<0
\end{cases}
$$

In contrast, traditional activation functions, such as logistic or tanh activation, have more
complicated derivatives. As throughout training typically thousands of gradients need to be 
computed this may add up to a substantial increase in running time. This is a reason,
why ReLU is nowadays preferred over logistic activation or tanh.


````{dropdown} How expensive is it to train a neural network?
:color: success
:icon: telescope

Training a large neural network is time consuming and requires substantial specialised hardware.
Often the computing power to train a large network is bought from a cloud computing 
provider such as AWS, Google or Microsoft Azure. How much does that cost? 
That is difficult to say. A study of Stanford University reports a cost of below \$10
(and training time below 5mins)
for an image recognition task and a network (ResNet50) that has about 25 million parameters. 
In contrast, for natural language recognition, a paper of Sharir et al. reports costs
between \$2500 and \$50000 for a 110 million parameter model. 
Training of ChatGPT-3 is estimated to have cost more than \$4 million, while Sam Altman claimed
that the training of ChatGPT-4 resulted in costs of more than \$100 million.  
Increasingly, only the largest 
companies can afford to train the most advanced networks. (OpenAI, the maker of ChatGPT, is backed by
Microsoft.)

[Stanford benchmark](https://dawn.cs.stanford.edu/benchmark/)<br/>
*The cost of training NLP models*, O. Sharir et al. (2020), [arxiv:2004.08900](https://arxiv.org/abs/2004.08900)<br/>
*AI’s Smarts Now Come With a Big Price Tag*, [Wired](https://www.wired.com/story/ai-smarts-big-price-tag/) (2021)<br/>
*ChatGPT and generative AI are booming, but...*, [CNBC](https://www.cnbc.com/2023/03/13/chatgpt-and-generative-ai-are-booming-but-at-a-very-expensive-price.html) (2023)
% https://dawn.cs.stanford.edu/benchmark/
% below $10 for ResNet50, which seems to have 23 million parameters

%$2.5k - $50k (110 million parameter model)
%$10k - $200k (340 million parameter model)
%$80k - $1.6m (1.5 billion parameter model)
% The cost of training NLP models,
% O Sharir, B Peleg, Y Shoham, arxiv:2004.08900, 2020

% 100 mill figure: https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/
````



The loss function
-----------------
% source: Michael Nielsen http://neuralnetworksanddeeplearning.com/chap3.html

Historically neural networks were trained with the square loss. Indeed, in a lot of applications
we are interested in the mean square error, so why not use it for neural networks, too?
It turns out that square loss might lead to slow learning -- in classification.

Recall that the square loss is defined as 

$$
\frac{1}{|S|}\sum_{(x,y)\in S}||f^{(K)}(x)-y||^2,
$$
 
so that the loss for an individual data point would be

$$
L_{(x,y)} = |f_1^{(K)}(x)-y|^2
$$

Here, we assume binary classification and thus a single output node. Note also that $y\in\{0,1\}$. 

We have already computed, see {eq}`sqbw`, the gradient at the output layer as 
\begin{align*}
\frac{\partial L_{(x,y)}}{\partial w^{(K)}_{1i}}
 & = 2( f^{(K)}_{1} -y )\cdot f^{(K)}_1(1-f^{(K)}_1) \cdot f^{(K-1)}_{i} \\
\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{1}} & = 2( f^{(K)}_{1} -y )\cdot f^{(K)}_1(1-f^{(K)}_1) 
\end{align*}
Mastering a skill is much harder than learning the first steps.
Intuitively, learning should be fast when we're very wrong, and much slower when we're almost
right.  
Strikingly, this is not what is happening here. 

Being very wrong would mean that $y=1$ and $f^{(K)}_1\approx 0$, or that $y=0$ and $f^{(K)}_1\approx 1$.
Then, the first factor in the equations above, $( f^{(K)}_{1} -y )$, would be close to 1 or -1, which is good.
The second term, however, $ f^{(K)}_1(1-f^{(K)}_1)$ would be close to 0, which is bad. 
The gradient would be very small, and learning would be slow; see {numref}`slowfig`. 
This can be observed in practice.[^Niel2]

[^Niel2]: At [Michael Nielsen's website](http://neuralnetworksanddeeplearning.com/chap3.html) there is a nice animation that illustrates this behaviour.

```{figure} pix/slow.png
:name: slowfig
:height: 6cm

Gradient $\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{1}} $ for square loss.
We assume that $y=0$. Marked: gradient when the network is very wrong.
```

*Cross entropy* as a loss function fixes this. In general, cross entropy 
measures a sort of distance between two probability distributions $p$ and $q$:

$$
H(p,q)=-\sum_{\omega}p(\omega)\log q(\omega)
$$
 
As  loss function in binary classification, cross entropy takes the following form

$$
L=-\frac{1}{|S|}\sum_{(x,y)\in S}y \log(h(x)) + (1-y)\log(1-h(x)),
$$

where $h:\mathcal X\to[0,1]$ is the binary classifier. 
Note here, that infinite loss may occur if $h(x)=0$ or $h(x)=1$. In practice, however, 
classifiers will seldom return $0$ or $1$, as there will always be some uncertainty. 
Note, moreover, that the loss is never negative.

In the setting of a single output 
neural network cross entropy loss turns into:

$$
L=-\frac{1}{|S|}\sum_{(x,y)\in S}y \log(f^{(K)}_1(x)) + (1-y)\log(1-f^{(K)}_1(x)),
$$

and, for a single fixed datapoint $(x,y)$ in the training set, we get

$$
L_{(x,y)}=-(y \log(f^{(K)}_1(x)) + (1-y)\log(1-f^{(K)}_1(x))).
$$



Again, we compute the gradient at the output layer; see {eq}`deltaK` and {eq}`gradb`,{eq}`gradw`.
We start with
\begin{align*}
\frac{\partial L_{(x,y)}}{\partial f^{(K)}_{1}}
= -\frac{y}{f^{(K)}_1} + \frac{1-y}{1-f^{(K)}_1} = \frac{f^{(K)}_1-y}{f^{(K)}_1(1-f^{(K)}_1)}
\end{align*}
We also recall that the derivative of the logistic function, the activation at the output layer, has a peculiar form: 
$\sigm'(z)=\sigm(z)(1-\sigm(z))$. We get
\begin{align*}
\frac{\partial L_{(x,y)}}{\partial b^{(K)}_{1}} & = \frac{\partial L_{(x,y)}}{\partial f^{(K)}_{1}}\cdot
\sigma'(g^{(K)}_1)
=  \frac{f^{(K)}_1-y}{f^{(K)}_1(1-f^{(K)}_1)} \cdot f^{(K)}_1(1-f^{(K)}_1)\\
& = f^{(K)}_1-y
\end{align*}
In the same way, we get

$$
\frac{\partial L_{(x,y)}}{\partial w^{(K)}_{1i}} = (f^{(K)}_1-y)f^{(K-1)}_i
$$

We observe that the counter-intuitive behaviour of square loss has vanished:
if the neural network is very wrong, ie, if $f^{(K)}_1$ and $y$ are very different,
then the gradient will be (comparatively) large. Again, this can be seen 
in real networks.[^losscomp]

[^losscomp]: {-} [{{ codeicon }}loss_compare](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/neural_networks/loss_compare.ipynb)

(softmaxsec)=
Softmax and loss
----------------
% nice illustration of softmax and log likelyhood at
% https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
% see also
% https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

The preceding discussion applied to binary classification and a single output 
node. What do we do when we have more classes? First, we'll have more outputs, namely
one for each class. Second, we'll use a softmax layer at the output. 

Assume we do image classification
with three classes: cat, pumpkin and bird. Then there will be three output neurons.
If a training sample $(x,y)$ shows a cat, then the class vector $y$ will be $y=(1,0,0\trsp )$.
If the image shows a pumpkin, we'll have $y=(0,1,0\trsp )$, and if it's a bird then
$y=(0,0,1\trsp )$. This way of encoding the classes even has a fancy name: *one-hot encoding*,
because there is always just one bit that is *hot*, ie, equal to 1.    

A softmax layer outputs a probability distribution over the classes. 
That is, each output node $f^{(K)}_k$ yields a value in $[0,1]$ and the sum over all
outputs equals 1.
So, an output might be $(0.2,0.7,0.1\trsp )$, which we interpret as follows: 
with 20\% confidence the image shows a cat, with 70\% confidence it's a pumpkin, and 
with 10\% it's a bird.


The softmax layer works a little bit differently than the activation functions we've seen
so far, as it is not a *scalar* function but a vector-valued function operating on 
vectors:

$$
\softmax:\mathbb R^{n_{K}}\to\mathbb R^{n_{K}}, g^{(K)}\mapsto f^{(K)}=\softmax(g^{(K)}),
$$

where the $k$th entry is equal to

$$
\softmax(g^{(K)}_k)=\frac{e^{g^{(K)}_k}}{\sum_{j=1}^{n_K}e^{g^{(K)}_j}}
$$

```{figure} pix/soft.png
:name: softfig
:width: 15cm

How a softmax output layer works
```

For a final softmax layer, the *cross-entropy loss* is normally used, and then takes this form:

$$
L=-\frac{1}{|S|}\sum_{(x,y)\in S}\sum_{k=1}^{n_K}y_k\log \left(f^{(K)}_k(x) \right)
$$

In the same way that cross entropy prevents slow learning for a logistic output layer, 
it also prevents slow learning a final softmax layer. 


%NOTE from Data Science and Machine Learning, Kroese et al
% EXERCISE?
Softmax can lead to over- or underflow (or numerical instability) if one of ${e^{g^{(K)}_k}}$
is very large, or if the sum $\sum_{k=1}^K{e^{g^{(K)}_k}}$ is very small. Fortunately, 
softmax is invariant under addition of a constant:

[^onevec]
```{math}
\softmax(g) = \softmax(g+\lambda \cdot 1)
```

[^onevec]: {-} What is the $1$ doing after $\lambda$? That's a vector with a 1 in every entry.

Thus, a dominant $g^{(K)}_k$ may be dealt with by substituting

$$
\softmax(g^{(K)})= \softmax(g^{(K)} - \max_{k} g^{(K)}_k\cdot 1)
$$

For regression tasks the activation function of the final layer is often 
simply the identity. We talk of a *linear output layer*. It turns out
that, for a linear output layer, the square loss is actually appropriate.
Slow learning, as in the case of a logistic output layer, does not occur.


Why cross-entropy loss?
-----------------------

Cross-entropy loss seems to make sense and certainly increases the more bad predictions are made -- still 
it may appear a bit arbitrary. It's not. We take a step back and figure out where cross-entropy loss is coming from.

*Cross-entropy* and entropy more generally is a concept from information theory. Cross-entropy, in information 
theory, compares two probability distributions $p,q$: 

$$
H(p,q)=-\expec_p[\log q]
$$

(I am a bit vague here, as in other places too, as to what kind of logarithm I mean. The short answer: there's both, the natural logarithm
as well base 2. As both variants only differ by a constant factor, it does not matter much, which one it is. If you press me on the base, 
I will claim it's base 2.)

More concretely, if $p,q$ are discrete probability functions on a sample space $\Omega$ then 

$$
H(p,q)=-\sum_{\omega\in\Omega}p(\omega)\log q(\omega)
$$


What is the connection to cross-entropy loss as used in machine learning? To keep things simple, let's concentrate
on binary classification. Then, for a classifier $h:\mathcal X\to [0,1]$, this could be a neural network with a single 
output and logistic activation at the output, the cross-entropy loss over the training set $S\subseteq \mathcal X\times\{0,1\}$
would be 

$$
L=-\frac{1}{|S|}\sum_{(x,y)\in S}y\log(h(x))+(1-y)\log(1-h(x))
$$
 
Recall that we model taking the samples for the training set with a probability distribution on $\mathcal X\times\{0,1\}$.
Earlier, that was often called $\mathcal D$; let's call that $p$ here. Moreover, let's define a conditional probability as

$$
q(y|x)=\begin{cases}
h(x) & \text{ if $y=1$}\\
1-h(x) & \text{ if $y=0$}
\end{cases} \quad\text{for all }(x,y)\in\mathcal X\times\{0,1\}
$$

We may interpret $q(y|x)$ as the probability that, given $x$, our classifier $h$ predicts class $y$. 
We use $q(y|x)$ rightaway to rewrite the cross-entropy loss as
\begin{equation}\label{ceq}
L=-\frac{1}{|S|}\sum_{(x,y)\in S}\log(q(y|x))
\end{equation}

We also simply write $p(x)$ for the *marginal probability*

$$
p(x) = p((x,0))+p((x,1))\quad \text{for all }x\in\mathcal X
$$



What have we accomplished? We can now see that the cross-entropy loss $L$ is a Monte-Carlo estimator[^MonteC] for $H(p,q)$.
For this, write $S=((x_1,y_1),\ldots, (x_m,y_m))$. Then
\begin{align*}
\expec_{S\sim p^m}[L] & = \expec_{(x_1,y_1)\sim p}\ldots\expec_{(x_m,y_m)\sim p}[L] \\
& = -\frac{1}{|S|} \sum_{i=1}^m\expec_{(x_i,y_i)\sim p}[\log q(y_i|x_i)] \\
& = -\frac{1}{|S|} \sum_{i=1}^m\expec_{(x,y)\sim p}[\log q(y|x)] \\
& = -\expec_{x\sim p}\expec_{y\sim p(\cdot |x)}[\log q(y|x)]  = \expec_{x\sim p}\left[ H(p(\cdot|x),q(\cdot|x))\right]
\end{align*}
(Here, I write $\expec_{x\sim p}$ to denote the expectation with respect to drawing $x$ with marginal probability $p(x)$.)

[^MonteC]: {-} What is a *Monte-Carlo* estimator? That is just fancy speak for: Let's take the mean of random samples.

As consequence, when we minimise cross-entropy loss (by searching for a better classifier $h$), we minimise, by proxy, 

$$
\expec_{x\sim p}\left[ H(p(\cdot|x),q(\cdot|x))\right]
$$

where we range over different $q$. Why is that good? First, that quantity will be small, if for each $x$
the quantity $H(p(\cdot|x),q(\cdot|x))$ is small. Why would we want that?

Fix an $x\in\mathcal X$.
We note that 
\begin{align*}
H(p(\cdot|x),q(\cdot|x)) & = -\expec_{p(\cdot |x)}[\log q(\cdot|x)] + \expec_{p(\cdot |x)}[\log p(\cdot|x)] -\expec_{p(\cdot |x)}[\log p(\cdot|x)]\\ 
& = \underbrace{\expec_{p(\cdot |x)}\left[\log\frac{p(\cdot|x)}{q(\cdot|x)}\right]}_{=:\KL(p(\cdot|x)||q(\cdot|x))} 
+ \underbrace{\left(-\expec_{p(\cdot |x)}[\log p(\cdot|x)]\right)}_{=: H(p(\cdot|x))}
\end{align*}
The second summand $H(p(\cdot|x))$ is called the *entropy* of $p(\cdot|x)$ and crucially does not depend on $q$, or on the classifier $h$. 
The first summand is called the *Kullback-Leibler divergence*. 
The Kullback-Leibler divergence is a measure of how similar $p(\cdot|x)$ and $q(\cdot|x)$ are. In particular, as we will see below, 
the divergence is always non-negative, and it is only $0$ if $p(\cdot|x)$ and $q(\cdot|x)$ are basically identical. 

What does that mean? If we minimise the cross-entropy loss, then, by proxy, we minimise $H(p(\cdot|x),q(\cdot|x))$ and that is precisely the 
case when we minimise $\KL(p(\cdot|x)||q(\cdot|x))$. Thus, we force $q(\cdot|x)$, which determines which class $h$ assigns to $x$, 
to become more and more similar to $p(\cdot|x)$. Why is that good? Because 

$$
x\mapsto \argmax_{y\in\{0,1\}} p(y|x)
$$

is the Bayes classifier, the best classifier achievable, and if $q(\cdot|x)=p(\cdot|x)$ for every $x\in\mathcal X$ 
then the classifier $h$ will be identical to the Bayes classifier.

We now finish with our discussion of cross-entropy loss. In the next section, we check that the Kullback-Leibler is indeed
a meaningful measure of how similar two probability distributions are.

````{dropdown} Information theory
:color: success
:icon: telescope

```{figure} pix/information_small.png
:width: 6cm
:align: left

midjourney on information
```
How can we measure information? That is a key question in information theory. 
One way to quantify information lies in compression: If some piece of data 
can be losslessly compressed to a small size then it only contains little
information, but if even the best compression algorithm still returns a large
file then information content will be high.

To make this slightly more formal, consider a text message in English, and assume $p$
to be the probability distribution on the words. So, $p$("money") will 
be comparably large but $p$("altruism") will likely be small. Shannon's 
source coding theorem now asserts that the best compression algorithm
will need, on average, about $H(p)$ many bits per word, where $H(p)$ 
is the entropy of $p$.
In this way, the entropy $H(p)$ can be seen to measure the average information
content in an English word. 
 (I am ignoring here the dependencies
between the words in a text. For example, the word "hamster" rarely follows the word "cheese".)

Cross-entropy $H(p,q)$, by the way, is the average compressed size of a word if the word
probability distribution is $p$ but I mistakenly think that it is $q$ and tailor the compression
algorithm accordingly to be optimal for $q$.


*Information Theory, Inference, and Learning Algorithms*, 
David J.C. MacKay (2003), [link to pdf](https://www.inference.org.uk/itprnn/book.pdf)
````

(KLsec)=
Kullback-Leibler divergence
---------------------------

The *Kullback-Leibler divergence* measures how different two probability distributions are. 
For the definition let us consider two discrete probability distributions $p$ and $q$ 
(the general case is similar but involves integrals and measure-theoretic caveats). Then 
the  Kullback-Leibler divergence between $p$ and $q$ is defined as 

```{math}
:label: KLdef

\KL(p||q)=\sum_{x} p(x) \log\left(\frac{p(x)}{q(x)}\right)
= \expec_p[\log p(x)] - \expec_p[\log q(x)]
```
Here, the sum ranges over all events; and some authors take the natural logarithm while others take the logarithm base 2. 
(As the difference is a constant factor, it does not matter much which logarithm is chosen.)
Moreover, we interpret $0\cdot \log 1/0$ as $0$, so that $\KL(p||q)$ is defined whenever $q(x)=0$ implies $p(x)=0$.

The Kullback-Leibler divergence behaves a bit like a metric:
it is always non-negative and only zero if $p=q$ (see below). 
However, it is not a metric as it does not satisfy the triangle inequality and as it is not even symmetric.
That is, normally $\KL(p||q)\neq\KL(q||p)$.

We consider a very simple example. Let $p$ be the probability distribution of a fair coin toss 
(ie, heads with probability a half and tails equally with probability a half), and denote by $q$
the probability distribution in which heads occurs with probability $\nu$. 
Then (taking the logarithm base 2)
\begin{align*}
\KL(p||q) & = \underbrace{\tfrac{1}{2}\log\left(\tfrac{1}{2}/\nu\right)}_{\text{heads}}
+ \underbrace{\tfrac{1}{2}\log\left(\tfrac{1}{2}/(1-\nu)\right)}_{\text{tails}} \\
& = \tfrac{2}{2}\log\left(\tfrac{1}{2}\right)- \tfrac{1}{2}\log \nu - \tfrac{1}{2}\log (1-\nu) \\
& = -1 - \tfrac{1}{2}\log\left(\nu-\nu^2\right)
\end{align*}
We immediately see that $\KL(p||q)=0$ if $\nu=\tfrac{1}{2}$. 

We also calculate $\KL(q||p)$:
\begin{align*}
\KL(q||p) & = \underbrace{\nu\log\left(2\nu\right)}_{\text{heads}}
+ \underbrace{(1-\nu)\log\left(2(1-\nu)\right)}_{\text{tails}} \\
& = \ldots = 1+\nu \log\nu + (1-\nu)\log(1-\nu)
\end{align*}
Evidently $\KL(p||q)\neq\KL(q||p)$ unless $\nu=\tfrac{1}{2}$. 

```{figure} pix/KL.png
:name: KLfig
:width: 15cm

Comparison of Kullback-Leibler divergence for varying values of coin toss probability $\nu$. 
Fair coin toss denoted by $p$, while under $q$ head occurs with probability $\nu$.
```

Let us prove *Gibb's inequality*:
% Proof from wikipedia

```{prf:Theorem}
:label: gibbsineq

Let $p,q$ be two (discrete) probability distributions. Then 
it holds that $\KL(p||q)\geq 0$, and 
$\KL(p||q)=0$ if and only if $p=q$.

```
Note: The theorem still holds for general probability distributions, with the only difference
that a Kullback-Leibler divergence of zero only implies that $p=q$ almost everywhere.

````{prf:Proof}
The proof rests on the observation that $z\mapsto -\log z$ is a convex function.
Thus, we may apply Jensen's inequality.
\begin{align*}
\KL(p||q) & = \sum_x p(x) \left(-\log\left(\frac{q(x)}{p(x)}\right) \right) 
 \geq -\log\left(\sum_x p(x) \frac{q(x)}{p(x)} \right) \\
 & = -\log\left(\sum_x q(x) \right) = -\log 1 = 0
\end{align*}
Indeed, $z\mapsto-\log z$ is even strictly convex, which means that 
the inequality is strict unless $q(x)/p(x) = q(x')/p(x')$ for all $x,x'$.

Assume that $c=q(x)/p(x)$ for all $x$. Then

$$
1=\sum_x q(x) = \sum_x cp(x) = c,
$$

and thus $q(x)=p(x)$ for all $x$. Therefore $\KL(p||q)=0$ implies $p=q$;
the other direction is trivial.
````


Local minima
------------

Neural networks are trained with SGD, or with a variant of SGD. 
Why does that work at all? The loss functions is highly
non-convex. Indeed, because of the symmetries in the weights it is almost guaranteed that 
there will be many local minima: permuting the nodes of a layer (together with their weights)
will turn one local minimum into another one. 

Why doesn't SGD regularly become trapped in a local minimum that is far away from the 
global minimum?  
Why are local minima not a problem? 


Empirically it is often observed that  local minima of large loss are rare. That is, 
 local minima often have costs that are close to the global minimum. Indeed, 
saddle points and plateaus, where the loss does not change much, seem to be more 
of an issue. We will try to give some theoretical justification for this 
observation. We will see two pieces of evidence; both will need a leap of faith. 

First, there is the observation that in random functions of many variables
critical points are usually saddle points. Indeed, a random polynomial in one variable will 
likely have local minima but for one in 12 variables this is unlikely. Let's look at a very simple
toy problem, a quadratic polynomial in one or two variables. 

Clearly, in one variable, the quadratic polynomial $x\mapsto ax^2+bx+c$ will have a local minimum 
(that is also a global one) as long as $a>0$. 
Thus, if we pick $a,b,c$ according to a normal distribution with mean 0 and variance 1
then the expected number of local minima is $\tfrac{1}{2}$.

Now, let's consider two variables:

$$
p:\:(x_1,x_2)\mapsto a_{11}x_1^2+a_{22}x_2^2+a_{12}x_1x_2+b_1x_1+b_2x_2+c
$$

We first compute the points with zero gradient:
\begin{align*}
&0 =\nabla p(x)=\twovec{\partial p(x)/\partial x_1}{\partial p(x)/\partial x_2} = 
\twovec{2a_{11}x_1+a_{12}x_2+b_1}{2a_{22}x_2+a_{12}x_1+b_2} \\
\Leftrightarrow  & \begin{pmatrix}2a_{11} & a_{12}\\a_{12} & 2a_{22} 
\end{pmatrix}\twovec{x_1}{x_2} = \twovec{-b_1}{-b_2}
\end{align*}
If the matrix is non-singular, which happens unless $4a_{11}a_{22}-a_{12}^2=0$, 
then there is therefore only one point with zero gradient. (And if we choose the coefficients
again with a normal distribution then the probability that the matrix is singular is 0.)

When is the single critical point a local minimum? When the Hessian is positive definite, 
ie, when its eigenvalues are positive. The Hessian is

$$
\begin{pmatrix}2a_{11} & a_{12}\\a_{12} & 2a_{22} 
\end{pmatrix}
$$

with eigenvalues determined by

$$
\lambda=a_{11}+a_{22}\pm \sqrt{(a_{11}-a_{22})^2+a_{12}^2}
$$

Let's again choose the coefficients according to a normal distribution with mean 0
and variance 1. Then a quick numerical simulation indicates that the probability
that both eigenvalues are positive is about 0.176. Thus, the expected number 
of local minima is $\approx 0.176$, which is already quite a bit smaller than the $\tfrac{1}{2}$
for one variable. 

With a slightly different model for a random polynomial, Dedieu and Malajovich[^Ded]
proved:

[^Ded]: *On the number of minima of a random polynomial*, J.-P. Dedieu and G. Malajovich (2008)


```{prf:Theorem} Dedieu and Malajovich (2008)
The expected number of extremal points of a random polynomial of degree d in $n$ variables 
is bounded by $\bigO(\exp(-n^2+n\log(d))$.
```
In particular, for large $n$ (and $\log d\ll n$) we do not expect *any* local minima or maxima.

What can we deduce for the loss landscape of a neural network? Obviously, the loss 
function of a neural networks is not a polynomial (piecewise polynomial, though!). 
And, more importantly, the loss is non-negative, which makes it implausible to think 
that the eigenvalues of the Hessian behave in a random manner when the loss is close
to 0: there is simply not much way to go down. Thus, there may be local minima! 
This
 argumentation, however, only works for small loss. Local minima
of small loss are not a problem, though: A neural network with small loss is, after all, 
precisely what we want to compute. 
If the loss is larger then it's not such a
stretch to think that the eigenvalues of the  Hessian might behave in a somewhat random 
way, which would make local minima unlikely. Admittedly, this argumentation 
involves a lot of hand-waving.


No traps
--------


Let's try again to argue that local minima are unlikely to trap 
SGD when training a neural network.
In the setting we will consider, there are more weights, more parameters of the 
neural network, than points in the training set. 
This is not unusual. Alexnet, a well-known 
neural network for image recognition had about 60 million parameters but was trained
on just 1.2 million datapoints.[^notraps]

[^notraps]: {-} [{{ codeicon }}no_traps](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/neural_networks/no_traps.ipynb)


```{figure} pix/locmin.png
:name: trainvarfig
:width: 15cm

Distribution of the training error for 100 runs of SGD in one hidden layer neural networks of 
different sizes. All neural networks were trained on Fashion MNIST with the same training set of size 5000.
For over-parameterised 
networks (100 hidden nodes) training error shows smaller variance, while the variance
is larger for fewer hidden nodes. Models were trained until training loss no longer improved.
```

Normally, such a setup is in danger of overfitting. Empirically, this does not seem 
to be that much of a problem, and in case, there are a number of precautions against 
overfitting that are routinely taken. I will just name-drop two, *early stopping*
and *dropout layers*, without explaining what they are.

An over-parameterised neural network can learn the training set perfectly. Thus, 
to see that a local minimum is also a global one, we need to check that it has 
zero training error. We will succeed in doing so, provided an extra condition is imposed
that might seem artificial and perhaps too far removed from reality. 
A similar phenomenon can be observed in practice: In {numref}`trainvarfig`, we see that 
over-parameterised neural networks are less likely to finish training with a training
error that is far away from the optimum.

% No bad minima: Data independent training error guarantees for multilayer neural networks
% D.~Soudry and Y.~Carmon (2016)

For two matrices $A\in\mathbb R^{m\times n}$ and $B\in\mathbb R^{k\times \ell}$ 
the *Kronecker product* $A\otimes B$ is defined as the matrix 

$$
A\otimes B=\begin{pmatrix}
A_{11}B & A_{12}B & \ldots & A_{1n}B \\
A_{21}B & A_{22}B & \ldots & A_{2n}B \\
\vdots & \vdots & \ddots & \vdots \\
A_{m1}B & A_{m2}B & \ldots & A_{mn}B 
\end{pmatrix}\in\mathbb R^{mk\times n\ell}
$$


The Kronecker product is the building block for a different product of matrices, the 
*Khatri-Rao product*. For this, let $A,B$ have the same number of columns, 
ie, $A\in\mathbb R^{k\times N}$ and $B\in\mathbb R^{\ell\times N}$. Then 
the Khatri-Rao product is the matrix of size $k\ell\times N$ defined as

$$
A\circ B=(A_{\bullet,1}\otimes B_{\bullet,1},\ldots,A_{\bullet,N}\otimes B_{\bullet,N})
$$

(Technically, this here is a special form of the Khatri-Rao product -- the general 
form operates on any two compatible block structures of $A$ and $B$.)

Let $\phi(x)$ be some statement that depends on $x\in\mathbb R^n$. For example,
$\phi(x)$ could be "the norm of $x$ is $||x||\neq 42$". Then we say that $\phi(x)$
holds for *almost all* $x\in\mathbb R^n$ if the set of exceptions

$$
\{x\in\mathbb R^n:\phi(x)\text{ false}\}
$$

has Lebesgue-measure zero, ie, is a null set.


% This is Lemma 13 from \emph{Identifiability of parameters in latent structure models with many observed variables},
% Allman, Matias and Rhodes
% https://arxiv.org/abs/0809.5032


The following lemma is due to Allman et al.[^All]
```{prf:Lemma} Allman, Matias and Rhodes (2009)
:label: otimeslem
Let $k,\ell,N$ be integers with $k\ell\geq N$. Then 
for almost all $(A,B)$ with 
$A\in\mathbb R^{k\times N}$ and $B\in\mathbb R^{\ell\times N}$
it holds for the Khatri-Rao product that

$$
\rank(A\circ B)=N
$$

```
[^All]: *Identifiability of parameters in latent structure models with many observed variables*, E.S. Allman, C. Matias, J.A. Rhodes (2009), [arXiv:0809.5032](https://arxiv.org/abs/0809.5032)

```{prf:Proof}
Note that, given $A,B$, the Khatri-Rao product $A\circ B$ has dimensions $k\ell\times N$. 
As $k\ell\geq N$, the matrix $A\circ B$ has not rank $N$ if and only if 
for every choice of $N$ rows from $A\circ B$ the matrix $M$ of these rows has determinant $\det M=0$.

Let us fix a choice of rows, namely the first $N$ rows of $A\circ B$.
The determinant $\det M$ is a polynomial in the entries of $A\circ B$, which in turn are poducts
of entries in $A$ and $B$. Thus, we can view $\det M$ as a multivariate polynomial $p(A,B)$
on the entries of $A,B$. Now the set of $(A,B)$ for which $A\circ B$ does not have rank $N$ is a 
subset of the set of roots of $p$

$$
\{(A,B):p(A,B)=0\}
$$

From analysis we know that the set of roots of a polynomial is a null set --- unless the polynomial 
is 0. Thus, let us prove that $p\not\equiv 0$.

To show that $p\not\equiv 0$ it suffices to find some $(A,B)$ for which $p(A,B)\neq 0$. 
Pick distinct primes $a_1,\ldots, a_k$ and $b_1,\ldots, b_\ell$ and consider the 
Vandermonde matrices

$$
A=\begin{pmatrix}
1 & a_1 & a_1^2 & \ldots & a_1^{N-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & a_k & a_k^2 & \ldots & a_k^{N-1}
\end{pmatrix}
\text{ and }
B=\begin{pmatrix}
1 & b_1 & b_1^2 & \ldots & b_1^{N-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & b_\ell & b_\ell^2 & \ldots & b_\ell^{N-1}
\end{pmatrix}
$$

From a beginner's linear algebra
course we may remember that a Vandermonde matrix, such as $A$, has full rank
unless two of the rows are identical (which can only happen if $a_i=a_j$ for some $i\neq j$).

Now, $A\circ B$ has the following form

$$
A\circ B = 
\begin{pmatrix}
1 & a_1b_1 & (a_1b_1)^2 &  \ldots &  (a_1b_1)^{N-1} \\
1 & a_1b_2 & (a_1b_2)^2 & \ldots & (a_1b_2)^{N-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & a_1b_\ell & (a_1b_\ell)^2 & \ldots & (a_1b_\ell)^{N-1} \\
1 & a_2b_1 & (a_2b_1)^2 &  \ldots &  (a_2b_1)^{N-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & a_kb_\ell & (a_kb_\ell)^2 & \ldots & (a_kb_\ell)^{N-1} 
\end{pmatrix}
$$

Obviously, $A\circ B$ is again a Vandermonde matrix.
By choice of $a_i,b_j$ as distinct primes, all products $a_ib_j$ are distinct, which means 
that the matrix consisting of the first $N$ rows (or indeed any choice of $N$ rows)
has non-zero determinant, ie, that $p(A,B)\neq 0$.
```

Given a neural network and a loss function $L$ over the training set, let us 
call a point $w$ (a collection of weights) a *differentiable local minimum*,
or *DLM*, if $L$ is differentiable at $w$, and if $w$ is a local minimum of $L$. 
Clearly, in a ReLU neural network $L$ is not everywhere differentiable but these
points are few (they form a null set) and it is at least plausible that most 
local minima in a typical neural network are differentiable. 

Let us consider a simple neural network with $n_0$ inputs,  a single hidden layer 
of $n_1$ nodes and a single 
output node. For technical reasons, we fix the activation function of the 
hidden layer to be leaky ReLU, let us say

$$
\textsf{leak}(z)=\begin{cases} z & \text{if }z\geq 0\\ 0.1z &\text{otherwise}\end{cases}
$$

and for the output we use the identity as activation, i.e.\ the output layer is a linear layer.
Let a training set be given that consists of $N$ datapoints $(x^{(n)},y^{(n)})$.

As usual, we denote the weights between input layer and hidden layer by $W^{(1)}$
and the weights between the hidden layer and the output layer by $W^{(2)}$.
For simplicity, we omit biases. 
On input $x^{(n)}$ we compute the input at the hidden layer as

$$
g^{(1)}(x^{(n)}) = W^{(1)}x^{(n)}
$$

Now, the output of the hidden layer then would be

$$
f^{(1)}(x^{(n)})=\textsf{leak}(W^{(1)}x^{(n)}) = \diag(a^{(n)}) \cdot W^{(1)}x^{(n)},
$$

where 

$$
a^{(n)}_h=\begin{cases}
1& \text{if }(W^{(1)}x^{(n)})_h\geq 0\\
0.1& \text{otherwise} 
\end{cases}
$$

The overall output of the neural network, on input $x^{(n)}$, then is

$$
W^{(2)}\diag(a^{(n)}) \cdot W^{(1)}x^{(n)}
$$


Now we do something that is non-standard and normally not part of the learning process. 
We perturb the activation function. That is, we replace $a^{(n)}$ by

$$
a^{(n)}_h = \epsilon_h^{(n)}\cdot\begin{cases}
1& \text{if }(W^{(1)}x^{(n)})_h\geq 0\\
0.1& \text{otherwise} 
\end{cases}
$$

where $\mathcal E=(\epsilon^{(1)}\ldots \epsilon^{(N)})$ is a matrix that should be thought of as
a small perturbation of the activation function.

We also collect all datapoints into a matrix $X=(x^{(1)},\ldots, x^{(N)})$,
and prescribe (mean) square loss as loss function
```{math}
:label: mselss
\text{MSE}=\frac{1}{N}\sum_{n=1}^N (y^{(n)}-W^{(2)}\diag(a^{(n)}) \cdot W^{(1)}x^{(n)} )^2
```
We now get to a main insight due to Soudry and Carmon:[^SC16]

[^SC16]: *Data independent training error guarantees for multilayer neural networks*, D. Soudry and Y. Carmon (2016), [arXiv:1605.08361](https://arxiv.org/abs/1605.08361)

```{prf:Theorem} Soudry and Carmon
:label: scthm

If $n_0n_1\geq N$ then all differentiable local minima of {eq}`mselss` are global
minima with $\text{MSE}=0$ for almost all $(X,\mathcal E)$.
```

The theorem does not imply that (almost) every local minimum is a global one. 
The additional assumption that we need in order to prove the theorem 
are simply to restrictive. However, the conditions are realistic enough 
that we plausibly may conclude that typically there few high training error local minima.

````{prf:Proof}
Consider {eq}`mselss` and observe that $W^{(2)}$ is a row vector as there is only one output node.
That means we can swap it with the diagonal matrix $\diag(a^{(n)})$ that succeeds it 
such that we get
\begin{align*}
\text{MSE} &=\frac{1}{N}\sum_{n=1}^N (y^{(n)}-W^{(2)}\diag(a^{(n)}) W^{(1)}x^{(n)} )^2\\
& =  \frac{1}{N}\sum_{n=1}^N (y^{(n)}-\trsp{(a^{(n)})}\diag(W^{(2)})  W^{(1)}x^{(n)} )^2
\end{align*}
We now write $W=\diag(W^{(2)}) \cdot W^{(1)}$ in order to simplify:

```{math}
:label: mselss2
\text{MSE} =\frac{1}{N}\sum_{n=1}^N (y^{(n)}-\trsp{(a^{(n)})} Wx^{(n)} )^2
```
We note that $(W^{(1)},W^{(2)})$ is a DLM of {eq}`mselss` if and only if $W$ is a DLM
of {eq}`mselss2`.

Now let's consider a DLM of {eq}`mselss2`. In particular, the gradient is 0:

$$
0=\nabla_W \text{MSE} = \frac{1}{N}\sum_{n=1}^N \nabla_W(y^{(n)}-\trsp{(a^{(n)})} Wx^{(n)} )^2
$$

Setting the error at $W$ to 

$$
e^{(n)}=y^{(n)}-\trsp{(a^{(n)})} Wx^{(n)}\in\mathbb R 
$$

we get for all $i,j$

$$
0=\frac{1}{N}\sum_{n=1}^N\frac{\partial}{\partial W_{ij}} (y^{(n)}-\trsp{(a^{(n)})} Wx^{(n)} )^2
=-\frac{2}{N}\sum_{n=1}^N  e^{(n)} a^{(n)}_i x^{(n)}_j
$$

This is the same as 

$$
0=\frac{2}{N}\sum_{n=1}^N e^{(n)} a^{(n)}\otimes x^{(n)} = \frac{2}{N} (A\circ X) e,
$$

for $A=(a^{(1)},\ldots,a^{(N)})$ and $e=\trsp{(e^{(1)},\ldots,e^{(n)})}$. 

If $\rank(A\circ X)=N$ then, as $e\in\mathbb R^{N}$, this would only be possible if $e=0$, that 
is, if MSE $=0$. Obviously, that means a perfect classification of the training set, and in turn,
that the local minimum is also a global one. 

Thus our task consists in showing that the set of $A\circ X$ with $\rank(A\circ X)<N$
is a null set. We cannot apply directly {prf:ref}`otimeslem` as $A$ implicitly depends on $X$:
indeed, whether $(W^{(1)}x^{(n)})_h\geq 0$ or $(W^{(1)}x^{(n)})_h<0$ determines whether 
$a^{(n)}_h=\epsilon^{(n)}_h$ or whether $a^{(n)}_h=0.1\epsilon^{(n)}_h$. However, for each $n,h$ there are 
only two options, which means that, given $\mathcal E$, there are only $2^{Nn_1}$ different possible matrices $A$. 

More precisely, there are $2^{Nn_1}$ different matrices $P$ with each entry either $0.1$ or $1$
such that $A=P\odot\mathcal E$, ie, such that $A$ is the Hadamard product of $P$ and $\mathcal E$. 
Now, if for some fixed such $P$ it holds that $\rank(A\circ X)=N$
for almost all $X,A$ then it also holds for almost all $X,\mathcal E$. Since the union of finitely
many null sets is still a null set, we deduce that for almost all $X,\mathcal E$ it holds that    
$\rank(A\circ X)=N$.
````

Soudry and Carmon extend the theorem to deep neural networks but the proof becomes more complicated.  

````{dropdown} What is the Kronecker product good for?
:color: success
:icon: telescope

```{figure} pix/blurry_Leopold_Kronecker_1865.jpg
:width: 6cm
:align: left

blurry Kronecker
```

A common challenge in image processing lies in recovering a sharp image from an out of focus, blurred photograph.
Such a *deconvolution* problem can in some situations be modelled as a 
matrix equation 

$$
AXB=I,
$$

where $A,B$ are matrices that represent the blurring, where $I$ is the blurred image and where $X$ is the unknown
sharp image. With the Kronecker product this can be rewritten as a simple system of linear equations

$$
(\trsp B\otimes A)\text{vec}(X)=\text{vec}(I),
$$

where $\text{vec}(\cdot)$ turns a matrix into a vector by stacking the columns on top of each other. 
This view then informs  theoretical insights. 

The Kronecker, or tensor, product also plays a prominent role in quantum computing. 

*Deconvolution and regularization with Toeplitz matrices*, P.C. Hansen (2002)<br/>
*An introduction to quantum computing*, N.S. Yanovsky (2007), [arXiv:0708.0261](https://arxiv.org/abs/0708.0261)

%The ubiquitous Kronecker product, Charles F. Van Loan
````

