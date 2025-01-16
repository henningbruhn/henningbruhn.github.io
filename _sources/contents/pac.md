$\newcommand{\bigO}{O}
\newcommand{\trsp}[1]{#1^\intercal} % transpose
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\DeclareMathOperator*{\vari}{\mathbb{V}}   % Probability
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\sigm}{\phi_{\text{sig}}} % logistic function
\newcommand{\bigOmega}{\Omega}
$


(pacsec)=
PAC learning
============

Let's imagine we urgently need a good classifier that distinguishes cat pictures from dog pictures.[^twoShai] 
How do we go about this? We perhaps decide that a neural network will be suitable and we 
prepare a training set: we collect cat and dog pictures and we label each picture as either 
a cat picture or a dog picture. Once we have done that we run an optimisation algorithm
to adapt the weights of the neural network so that the training error, the 
misclassification rate on the training set, is as small as possible. (We will discuss 
such optimisation algorithms in a later chapter.) What concerns us here is: 
Why can we expect a neural network with small training error to perform well on new data?

[^twoShai]: Much of the material is based on *Understanding Machine Learning* by Shai Shalev-Shwartz and Shai Ben-David

Recall: We do not care about the training set. We already know which picture in the training
set is a cat and which is a dog. What we care about is performance on new data, ie, 
we strive for a small generalisation error. 
Does a small training error *guarantee* a small generalisation error? 
In short: No. 

Here is an example with zero training error but large generalisation error. 
We imagine a binary classification task with domain set  $\mathcal X=[0,1]\times [0,1]$,
in which 
every point $x\in\mathcal X$ in the upper half
is in class 1, and every point in the (open) lower half is in -1, and every point in $\mathcal X$ is equally likely. 
That is, the *marginal* distribution over $\mathcal X$ is uniform. We use zero-one loss.
We draw a training set $S$. 

We can easily devise a classifier with *zero* training error:

(memalgo)=
```{math}

\text{Mem}(x)=
\begin{cases}
y &\text{if }(x,y)\in S\\
1 &\text{else}
\end{cases}
```

This algorithm *memorises* the training set and returns a default label, $1$ in this case, for 
datapoints outside the training set. Clearly, it is rubbish.

In fact, the Bayes error is 0 in our example (the class is fully determined by the domain set), 
the classifier Mem, however, only achieves a generalisation error of $\tfrac{1}{2}$. 
By {eq}`zorisk` we have:

$$
L_{\mathcal D}(\text{Mem})=\proba[x\text{ in upper half}]\cdot 0+\proba[x\text{ in lower half}]\cdot 1
=\tfrac{1}{2}
$$

Note, here, that taken over the infinite unit square, the finite sample $S$ has probability weight 0,
and so does not appear in the calculations. 


```{figure} pix/dec_overfit.png
:name: decoverfig
:width: 15cm

Datapoints in top half are class +$ with 90\% probability, 
in lower half they are class -1 with 90\% probability. Shown are 
four trained predictors, each time with training set and decision boundary. 
Top left: Neural network with one hidden layer of 10 neurons. Top right: Neural network 
with two hidden layers of 100 neurons each. Lower left: Decision tree of depth 1 (ie, a single decision rule). 
Lower right: Decision tree grown to full height (with zero training error). 
The simpler predictors, in the left column, show better generalisation than the more 
powerful ones in the right column.
```

The example clearly appears contrived and may be seen as cheating. 
[Mem](memalgo) is completely artificial. Nobody would use it in practice. 
Yet, similar outcomes may be observed with commonly used predictors. 
Consider a distribution with domain set $[0,1]^2$ that assigns the label 1 with probability $90\%$ 
to datapoints in the upper half, and label -1 with probability 90\% to points 
in the lower half.
In {numref}`decoverfig` two neural networks (top row) and 
two decision trees (bottom row) are fit to a training set 
from that distribution. The predictors in the left column are fairly simple, the ones in the 
right column are more powerful: The neural network on the left is set up to have one hidden layer
with 10 neurons, the one on the right has two hidden layers with 100 neurons each. The decision tree
on the left consists of a single decision rule (depth 1), while the one the right was allowed 
to grow to an arbitrary depth and thus can fit the training set perfectly. While 
the more powerful predictors on the right have smaller training error, they
have larger generalisation error than the simpler predictors on the left. 
We say that the predictors in the right column are *overfitting*: they adapt very well to the 
training set but learn less about the hidden distribution. 

What is the reason for overfitting? Overfitting may occur if the classifier has a large degree of freedom
compared to the size and complexity 
of the training set. What, however, is the degree of freedom of the classifier, how can 
we measure it and how does it impact the generalisation error? This is what we'll try to figure out in this chapter.


Empirical risk minimisation
---------------------------

How do we train a classifier? 
Given a domain set $\mathcal X$ and a set of classes $\mathcal Y$, and 
a (hidden) distribution $\mathcal D$ on $\mathcal X\times\mathcal Y$, we 
draw a training set $S$ of size $m$ from the distribution $\mathcal D$. 
We write $S\sim\mathcal D^m$ to denote that we draw $m$ samples from $S$
indepedently of each other.  Observe that it might happen that we draw a given point twice or even more 
often. 
How now should we choose a classifier $h:\mathcal X\to\mathcal Y$? It should have low training error.
Often, in this context, the training error is also called \defi{empirical risk}:

$$
L_S(h)=\frac{1}{|S|}\sum_{(x,y)\in S}\ell(y,h(x))
$$

(Here, and below, $\ell$ will be a loss function. When in doubt, assume that it is zero-one loss.)

In practice, we do not choose a classifier from the set of *all* possible 
classifiers. In fact, we would not know how to do that. Instead, we have an
algorithm to compute a decision tree, and we have a (different)
algorithm to train a neural network and so on. When we train a classifier we therefore
choose a classifier that is best, in some sense, within a *restricted* class. Perhaps 
we compute the best decision tree of depth at most 10, or the best linear classifier. 



Fix a set $\mathcal H$  of classifiers $h:\mathcal X\to\mathcal Y$. 
How do we now choose 
a classifier from $\mathcal H$? 
With the help of a suitable optimisation algorithm,
we choose a classifier $h_S$ in $\mathcal H$ with 
smallest training error (empirical risk) within the class $\mathcal H$:

```{math}
:label: ermparadigm
h_S=\argmin_{h\in\mathcal H} L_S(h)
```

This procedure is called *empirical risk minimisation* (ERM) paradigm.


Before we go on, a little bit of fineprint. I am neglecting here something: in practice, we do not 
necessarily find the perfect minimiser as in {eq}`ermparadigm` but will need to make do with an
approximation. In fact, we do not know how to efficiently compute the decision tree (of limited depth, say)
with smallest empirical risk, and even for linear classifiers we might need to contend ourselves
with a classifier that is only very close to being optimal. However, to keep things a bit simpler 
we will pretend for the moment that we can compute $h_S$ as in {eq}`ermparadigm`.


Below I will keep using the notation $h_S$ to denote the (or rather, *a*) classifier
returned by the ERM paradigm for a training set $S$. Let me stress that $h_S$ also depends 
on the loss function $\ell$ and on the set $\mathcal H$ of classifiers, even though $\ell$ and $\mathcal H$
do not appear in the notation $h_S$. (And indeed, $h_{S,\ell,\mathcal H}$ would really be quite
cumbersome.)

Test error and generalisation error
-----------------------------------

We have introduced the test set as a stand-in for new data. Let's see 
what the test error can tell us about the generalisation error.

We need a *measure concentration* inequality, an inequality that 
asserts that the mean of certain random variables is, with high probability,
very close to the expected value -- provided a number of mild conditions are satisfied.[^conccode]

[^conccode]: {-} [{{ codeicon }}concentration](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/pac_learning/concentration.ipynb)

```{prf:Theorem} Hoeffding's inequality
:label: hoeffthm
Let $\epsilon>0$, 
and let $X_1,\ldots, X_m$ be independent random variables with values in $[a,b]$
such that $\mu=\expec[X_i]$ for all $i=1,\ldots, m$. Then 

$$
\proba\left[\left|\tfrac{1}{m}\sum_{i=1}^mX_i-\mu\right|\geq\epsilon\right]\leq 2e^{-\frac{2m\epsilon^2}{(b-a)^2}}
$$
```
 
Consider a *fixed* classifier $h\in\mathcal H$ and draw a random sample $S$ from 
the hidden distribution on $\mathcal X\times\mathcal Y$:

$$
S=((x_1,y_1),\ldots, (x_m,y_m)).
$$

(This could be the training set, it could the test set or some other set.)
We can then, for $i=1,\ldots,m$, consider $X_i=\ell(y_i,h(x_i))$ as a random variable. 
Since the sample $S$ is drawn in *iid* fashion, the random variables $X_1,\ldots, X_m$
are independent and identically distributed. In particular, they all have the same expectation

$$
\expec[X_i]=\expec_{(x,y)\sim\mathcal D}[\ell(y,h(x))],
$$

which is nothing else than the true risk $L_{\mathcal D}(h)$! Moreover, we defined the $X_i$
such that their average

$$
\frac{1}{m}\sum_{i=1}^mX_i=L_S(h)
$$

coincides with the empirical risk.
Earlier, we had required of a loss function (in a classification task)
to take its values in $[0,1]$. Keeping this in mind,  we may apply Hoeffding's inequality
 to obtain:
 
 ```{math}
 :label: oneh

\proba\left[|L_\mathcal D(h)-L_S(h)|\geq\epsilon\right]\leq 2e^{-{2m\epsilon^2}}
```

As a consequence, for at least moderately large $m$,
the training error of $h$ will not differ much from the generalisation error. 
Does this also mean that the generalisation error of $h_S$, the minimiser in {eq}`ermparadigm`, is
close to its empirical risk (with high probability)? 
No -- this estimation holds only if $h$ is fixed *before*
we draw the sample $S$. The classifier $h_S$, however, depends on $S$.

The estimation {eq}`oneh` *can*, however, be directly applied to the 
*test* error, the average loss on the test set $T$. 
The classifier $h_S$ does not depend on $T$. Indeed, 
one of the holy rules in machine learning is that the test set cannot be used in any form
when learning the classifier. Therefore

```{math}
:label: testoneh
\proba\left[|L_\mathcal D(h_S)-L_T(h_S)|\geq\epsilon\right]\leq 2e^{-{2|T|\epsilon^2}}
```
The probability obviously becomes very small for large test set size.

Let us rewrite this as 

$$
\delta= 2e^{-2|T|\epsilon^2} \Rightarrow \epsilon=\sqrt{\frac{\ln(2/\delta)}{2|T|}}
$$

```{prf:Theorem}
:label: testerrthm

Let $h$ be some classifier, let $\delta>0$, and let $T$ be a test set. Then
with probability at least $1-\delta$ (over the choice of $T$):

$$
L_{\mathcal D}(h)\leq L_T(h)+\sqrt{\frac{\ln(2/\delta)}{2|T|}}
$$

```

In particular, the theorem holds for $h_S$, the classifier that minimises the empirical risk. 
As an illustration, with a probability of $95\%$ if we draw a test set of 10000 data points, 
then the test error will differ from the true risk by at most 1.4\%. [^testerrcode]
%Kaggle!

[^testerrcode]: {-} [{{ codeicon }}testerr](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/pac_learning/testerr.ipynb)


In {numref}`testerrvarfig` the variability of the test error is illustrated. A decision tree was trained
on the MNIST task. Then 100 disjoint test sets of size 500 each were drawn and their test error computed.
The test error shows a significant spread. While this is due to the size of the test set, which 
is clearly severely undersized, a simimlar phenomenon must also be anticipated for more reasonably sized 
test sets. In particular, when comparing different predictors by their test error, we need to check whether
a better test error of one algorithm actually means that the predictor is better, or whether 
the test error is simply a statistical fluctuation. 

How can this be checked? For instance, by repeating the comparison on a newly shuffled dataset. That is, 
from the whole dataset draw a new training and test set, retrain the algorithms, and redo the comparison,
and repeat this several times. Even better is to compare the predictors on a number of different datasets. 
%% variance of test error
```{figure} pix/testerr_var.png
:name: testerrvarfig
:width: 8cm

Test errors of a decision tree on the MNIST task. Training set size was 20000 samples.
The remaining data was partitioned into 100 test sets of size 500 each. The histogram shows the 
distribution of errors on these tests sets.
```

%% don't use the test error for model selection
Often in articles of the field one may see a comparison of a predictor developed in the article
with several predictors that were previously known. Usually, the authors then compute the test errors,
the new predictor comes out ahead, and the authors then argue that their algorithm is superior 
to the others. 
If done carefully, and
as long as the statistical fluctuation of the test error is taken into account there 
is not much wrong with this approach. This changes if such a comparison is used to pick the 
"best" predictor. 

Why is that? Imagine training a homogeneous linear classifier on data from $\mathbb R^2$, and 
assume that the machine precision is so low each of the two weights can only take one of 256 values
(ie, the weights are bytes). That means there are only $256^2=65536$ different such linear classifiers. 
If we now compare all those classifiers by their test error, then we effectively train a linear 
classifier on the test set as training set. The test set *becomes* the training set. 
What is the consequence? The test error is no longer a good estimator for the true risk. 

Clearly, this is an artificial example. It illustrates, however, a situation that may occur in 
practical applications. Neural networks, for instance, have many degrees of freedom. 
There are certainly the weights, that are adjusted during training, but also the whole architecture:
How many hidden layers does the neural network have? How many nodes in each layer? What 
activation function is chosen? The actual training, the optimisation process, introduces more 
decisions: Which optimisation algorithm is chosen? How are the parameters that govern the optimisation 
process chosen? 

Often the parameters other than the weights that are fixed during training
are called *hyperparameters*. As there are many possible settings for the hyperparameters
it is common to train many instantiations of the same predictor with different hyperparameters.
If then 
the best setting is chosen by comparison of the test errors, information on the test set 
leaches into the design of the predictor. As a consequence, the test error no longer 
faithfully approximates the true risk. This phenomenon can be observed in practice. [^selcode]

[^selcode]: {-} [{{ codeicon }}selection](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/pac_learning/selection.ipynb)


Therefore:
> don't use the test error for model selection!

How should model selection, the selection of the best predictor, be done instead? With a
third dataset, the *validation set*. Ideally, there is enough data to furnish a training
set, a validation set for hyperparameter tuning and model selection, and a test set 
for an unbiased estimate of the true risk.

Overfitting and underfitting
----------------------------

```{figure} pix/gengap.png
:name: gengapfig
:height: 8cm

More training data, smaller generalisation gap. MNIST data set, decision tree
with max depth fixed to 10.```


Given a training set $S$ and test set $T$ we can decompose the true risk of $h_S$ as follows:

```{math}
:label: geneq
L_{\mathcal D}(h_S)=\underbrace{\left(L_{\mathcal D}(h_S)-L_{T}(h_S)\right)}_{\text{small}}+
\underbrace{\left(L_{T}(h_S)-L_{S}(h_S)\right)}_{\text{generalisation gap}}+
\underbrace{L_{S}(h_S)}_{\text{training error}}
```

By {prf:ref}`testerrthm`, the first part can be seen to be reasonably small, and the last part is 
simply the training error, aka, the empirical risk. The difference between test and training error
is also called the \defi{generalisation gap}. A large generalisation gap
means that the classifier learns the training set and not the underlying distribution $\mathcal D$.
In other words, the classifier is *overfitting*. Often this is the case because the classifier
(or rather the class $\mathcal H$) has too many degrees of freedom.
If the training error is large then 
the classifier is not flexible enough to accomodate the training set --- it is said to *underfit*; see {numref}`underfitfig`. 


```{figure} pix/underfit.png
:name: underfitfig
:width: 12cm

The linear predictor on the left underfits the training set. The quadratic predictor on the 
right fits the training set perfectly.
```


Test and training error vary with increasing training set size. A typical relationship is shown 
in {numref}`gengapfig`: the generalisation gap decreases with larger traininig set size, while 
the training error increases.

Let me note, that some authors mean the difference

```{math}
:label: ggengap

L_{\mathcal D}(h_S)-L_{S}(h_S),
```
when they talk about the generalisation gap. This quantity is obviously of much interest:
we optimise against the training error but want a small generalisation error. 
However, we cannot compute {eq}`ggengap` directly but we can measure
$L_{T}(h_S)-L_{S}(h_S)$. As long as the test set is large enough it makes not much 
difference which of the two quantities we consider as they are very close to each other. 


