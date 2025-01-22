$\DeclareMathOperator{\sgn}{sgn}
\newcommand{\trsp}[1]{#1^\intercal} % transpose
\newcommand{\sigm}{\phi_{\text{sig}}} % logistic function
\newcommand{\twovec}[2]{\begin{pmatrix}#1\\#2\end{pmatrix}}
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\newcommand{\bayes}{h_{\text{Bayes}}} % the Bayes classifier
\newcommand{\bayerr}{\epsilon_\text{Bayes}} % the Bayes error
\DeclareMathOperator*{\argmax}{argmax}
$

Predictors, classification and losses
=====================================

Does a given picture show a dog or a cat?
Should the applicant be granted a credit? Is the mail
spam or ham? These are all examples of *classification* tasks, one of the 
principal domains in machine learning. 

A well-known classification tasks involves the MNIST data set.[^mnist]
The MNIST data set contains 70000 handwritten digits as images of 28$\times$ 28 pixels. The task is 
to decide whether a given image $x$ shows a 0, 1, or perhaps a 7. In machine learning
this tasks is solved by letting an algorithm *learn* how to accomplish the tasks by 
giving it access to a large number of examples, the *training set*,
together with the true classification.[^mnistcode] 
In the MNIST tasks that means that the algorithm not only
receives perhaps 60000 images, each containing a handwritten digit, but also for each image
the information which *class* it is, ie, which of the digits 0,1,...,9 is shown; see {numref}`mnistfig`.  

[^mnist]: {-} MNIST is so well known that it has a [wikipedia](https://en.wikipedia.org/wiki/MNIST_database) entry.
[^mnistcode]: {-} [{{ codeicon }}MNIST](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/MNIST.ipynb)\
(everywhere you see this icon, there's a link that takes you directly to colab)

```{figure} pix/mnist_row.png
:name: mnistfig
:width: 15cm

A small sample from the MNIST data set
```

  

Machine learning is part of the wider field of *artificial intelligence*. Early approaches to AI 
relied almost entirely on rules. For digit recognition, a number of rules could be formulated such as:

* if the digit contains an o-shape then it's 0,6,8 or 9
* if the digit contains a long vertical stroke then it's 1,4 or 7
* ... (many many more rules)

But how exactly do you identify an o-shape? That could be done by another rule. Sometimes, though, a 2
might be written in such a way that there's also a little "o" in the figure -- how do you cope with that?
Quite quickly hand-coded rules become very complicated. Ultimately, at least for tasks such as 
handwritten digit recognition a rules based approach is very tedious and not very 
successful.[^needto]

[^needto]: Here's a [comic essay](https://weneedtotalk.ai/) that explains the basics of modern AI. It's very nice. 

Machine learning pursues a different strategy: the algorithms are set up to improve, or learn, 
from experience (data). That is, a machine learning algorithm doesn't directly perform the 
required task but instead uses (large amounts of) data to *learn* the actual algorithm, the *predictor*, that 
then does the task. For example, for the MNIST task 
the machine learning algorithm called *random forest*
is fed many samples of handwritten digits together with the correct classification (this is a "7")
in order to devise the actual classification algorithm (that, confusingly, is also often called 
a random forest). 


```{figure} pix/learning.png
:name: learningfig
:width: 15cm

Learning an algorithm, the predictor
```



Consequently, machine learning typically works in two phases: the *learning or training phase*
and the *inference or prediction phase*. During the training phase the prediction algorithm is learnt -- this 
often takes a long time and requires expensive and extensive hardware. Then, in the prediction phase, we 
actually use the resulting predictor, ie, we point our smartphone at a menu in France and it seamlessly 
translates "crêpe" to "pancake". The prediction phase is often much less computationally expensive
than the training phase and might for some applications even work directly on your phone.

We will mostly be concerned with *supervised learning*: there, during the training phase, the learner
is fed the data together with the ground truth. That is, the input might consist 
of many pairs of a picture (as jpeg, say) and a single bit, where perhaps a 0 signifies that 
the picture shows a cat and a 1 that it shows a dog.
In the prediction phase the algorithm obviously is input only a picture and is expected to tell
whether the picture shows a cat or a dog.

There is also *unsupervised learning*:[^semi] a setting in which the learning algorithm does not 
have access to the (or some) ground truth during training.
We use unsupervised learning to detect
patterns in data or to learn some complicated probability distribution. 
So, for instance, an algorithm might learn how typical credit card transactions look like and
then flag anomalous transactions because they might be fraudulent.

[^semi]: {-} There's also semi-supervised learning and more forms of learning.

Tasks in ML
-----------
% Goodfellow et al

Some common tasks in machine learning are:

* *Classification:* classify some input into a finite number of classes. Examples are 
recognition of handwritten letters, facial recognition etc. 
* *Regression:* 
predict a numerical value from the input. This might be the price a house might fetch 
on the market, or the expected medical costs an insurance will need to pay out to a customer over the year. 
(Why the weird name? What is "regressing"
in linear regression, ie, 
"returning to a former or less developed state" according to Oxford Languages? The short answer: nothing.
The term apparently comes from the paper 
of Francis Galton (1886).[^Gal86])
%\item \emph{Transcription:} OCR, speech recognition, Google StreetView and address numbers
* *Machine Translation:* automatically translate from one language to another. 
%\item \emph{Structured Output:} image segmentation, image captions. 
* *Anomaly / novelty detection:* detect anomlies in signals; this could, for example, be a fraudulent credit card use. 
* *Imputation:* guess or deduce missing values in data.
* *Denoising:* remove noise from images, videos, sound recordings.
* *Recommender Systems:* recommend items, such as books, movies, music, to users based on their taste.
* *Reinforcement Learning:* train intelligent agents to take the right action depending on 
the situation. Think chess programs and robots.

There are more. A *generative AI*, for instance, may summarise research articles or produce images, sound or even movies. 

[^Gal86]: *Regression Towards Mediocrity in Hereditary Stature*, F. Galton (1886), [jstor](https://www.jstor.org/stable/2841583?seq=1)

We will first concentrate on classification and, to a lesser extent, on regression.

Vocabulary
----------

In a classification task, we aim to train a classifier that assigns labels or a class to 
given input data. 
In the MNIST digit recognition task, the input consists of grey-scale images of size 
28$\times$28 pixels. This is the *domain set*, 
the set $\mathcal X$ that contains all (possible) data points for the task at hand.
Normally $\mathcal X$ is a finite-dimensional vector space such as $\mathbb R^n$, or at least a 
subset of $\mathbb R^n$. In the MNIST task, we could set $\mathcal X=\{0,\ldots, 255\}^{28\times 28}$, 
if we assume that each pixel has a grey value in 0,...,255. 

The entries of a data point $x\in\mathcal X$ are
the *features* or *attributes* of $x$. This could be 
the grey value of a pixel or the income of a customer.

The aim is to predict a *label* or *class* for each data point $x\in\mathcal X$.
We denote the set of labels or classes by $\mathcal Y$. For MNIST, we have $\mathcal Y=\{0,1,\ldots, 9\}$.
For spam-detection, $\mathcal Y$ could be $\{0,1\}$, where 0 would perhaps mean that it's spam, and 
1 would indicate not-spam. If $\mathcal Y$ consists of only two classes, we talk of *binary classification*,
and then we usually take $\mathcal Y$ to be $\{0,1\}$ or $\{-1,1\}$.
In classification, $\mathcal Y$ should be finite and is often relatively small. For regression tasks
$\mathcal Y$ is usually infinite, and often equal to $\mathbb R$ or to some interval $[a,b]$.
There are also multidimensional regression problems, problems with $\mathcal Y=\mathbb R^n$.


To solve a classification task, we devise a *classifier*, a function $h:\mathcal X\to\mathcal Y$. 
The training classifier is obtained as a product of the *learning algorithm*, an algorithm that 
takes as input training data and outputs a classifier.  
Here, the *training set* is a finite subset of $\mathcal X\times\mathcal Y$.
For MNIST the tuples $(x,y)$ in the training set would consist of the image $x$ of a digit and $y\in\{0,\ldots, 9\}$, 
the digit shown in the image. That is, $y$ is the *true class* of $x$. (It should be noted, however, 
that there may be errors in the training set, and that sometimes the truth is ambiguous: 
of two credit applicants with the exact same financial data $x$ one may have been deemed credit worthy
and the other's application may have been rejected.)

When is a classifier good? This is a central question that needs a subtle answer that we'll defer to 
later. A non-subtle answer, however, would be: if it classifies many data points correctly. 
More generally, a classification or regression task comes with a *loss function*,
a function $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$ that fixes a penalty for 
misclassifying a data point. A very common loss function is the *zero-one loss*

$$
\ell_{0-1}(y,y')=
\begin{cases}
0 & \text{if }y=y'\\
1 & \text{if }y\neq y'
\end{cases}
$$

% loss function upper-bounded by 1 because of Hoeffding
A loss function should not penalise correctly classified data points.
That is, 
it should satisfy $\ell(y,y)=0$ for all $y\in\mathcal Y$. Moreover, for classification 
tasks we assume that $\ell(y,y')$ is upper-bounded[^upploss] by 1.

[^upploss]: Why? Mostly for technical reasons: We will apply Hoeffding's inequality later, where
the range of the loss function matters.

With a loss function, we can define the *training error*:
If $S\subseteq \mathcal X\times\mathcal Y$ is the training set of size $m$ then
the training error of the classifier $h$ is 

$$
L_S(h)=\frac{1}{m}\sum_{(x,y)\in S}\ell(y,h(x))
$$
 
This is simply the average of the loss function over the training set. 
For the zero-one loss, the training error is equal to the fraction of misclassified 
data points in the training set. 
Technically, 
the notation $L_S(h)$ for the training error of classifier $h$ would need to 
include the loss $\ell$ as well, as different loss functions will result in 
different training errors. However, $L_{S,\ell}(h)$ is too many symbols. 
If the loss is not clear from the context, I will specify it explicitly. 

````{dropdown} Examples of classification tasks
:color: success
:icon: telescope

OCR
: In *optical character recognition*, hand-written or printed text is transcribed. Applications are 
numerous. One of the earliest is automatic sorting of mail (of the paper kind).

iceberg monitoring
: Icebergs may endanger offshore oil rigs and container ships. Machine learning can help
to detect icebergs (and to differentiate them from ships and other features) on satellite data. 
% kaggle challenge

spam detection
: Given an email is it spam or not? 

fault detection
: Painting and coating cars is a complex and time-consuming process. It is important to 
detect coating faults as early as possible as late detection may result in higher cost remedies. 
During the process high-dimensional sensor data is recorded, based on which a classifier recommends 
certain auto bodies for closer inspection. This application was developed and implemented in a
Master's thesis here in Ulm, and is now used at a major car maker.

high energy physics
: The Large Hadron Collider at CERN produces every hour about as much data 
as Facebook collects in a year. The amount of data is much too large for human inspection. Instead a variety of machine learning algorithms
are employed: some throw out uninteresting data; others categorise the data by the type of
particle interaction.

Note the different types of data and labels. In the first two examples, the input consists of image data. Spam
detection is text-based. Fault detection in production processes is often based on time-series sensor data. 
The sensors could record electrical currents, magnetic fields or simple audio data.


Iceberg challenge, [kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)<br/>
*Machine learning at the energy and intensity frontiers of particle physics*, Radovic et al., Nature (2018) 
````


Linear predictors
-----------------
% 9.1
% 9.1.1
% 9.1.2 in Ben-David & Shwartz

Let's look at possible the simplest sort of classification algorithm, a *linear classifier*.

We consider a binary classification task with domain set $\mathcal X\subseteq \mathbb R^d$ and $\mathcal Y=\{-1,1\}$.
A very simple classifier tries to separate the data points into different halfspaces. That is, we look for a halfspace
$H=\{x\in\mathbb R^d: \trsp wx\geq 0\}$ and then classify all points in $H$ as $+1$, say, and all points 
in $\mathbb R^d\setminus H$ as $-1$. More generally, we could use an affine hyperplane defined by $w\in\mathbb R^d$
and $b\in\mathbb R$ and then classify as follows

$$
h(x)=\begin{cases}
+1&\text{if }\trsp wx+b\geq 0\\
-1&\text{if }\trsp wx+b<0
\end{cases}
$$

If we let $h_{w,b}$ be the affine function $x\mapsto \trsp wx+b$ then the classifier is defined as

$$
\sgn\circ h_{w,b},
$$

where $\sgn$ denotes the sign function, with the slightly non-standard convention that $0$ is attributed sign $+1$.
A classifier such as $\sgn\circ h_{w,b}$ is a *linear predictor* or *linear classifier*; see {numref}`linpredfig`. 
The term $b$ in $\sgn\circ h_{w,b}$ is the *bias* of the linear predictor.

```{figure} pix/sep_and_non_sep.png
:name: linpredfig
:width: 15cm

The left dataset can be fit perfectly by a linear classifier, the one on the right cannot.
```

By modifying the domain set slightly
we can even omit the bias. Indeed, define $\tilde{\mathcal X}$ as the set 

$$
\tilde{\mathcal X}=\left\{\twovec{x}{1}:x\in\mathcal X\right\}
$$

Then $h_{w,b}(x)=1$ if and only if 

$$
\tilde h_{\tilde w,0}\left(\twovec{x}{1}\right)=1,\, \text{where }\tilde w=\twovec{w}{b}
$$

The homogeneous case is often easier to handle.
Then, we will also write $h_w$ for $h_{w,0}$, that is

$$
h_w: x\mapsto \trsp wx
$$

%%In the homogeneous case, that $S$ is separable can be written more succinctly as: $y\cdot \trsp wx>0$ for 
%%all $(x,y)\in S$. As $S$ is finite this, in turn, is equivalent to  $y\cdot \trsp wx\geq 1$.
%%(Scale $w$!)


(logregsec)=
Logistic regression
-------------------

How can we train a linear predictor? Ideally, we would minimise the training error directly. 
Let's assume 
 a homogeneous training set, which means that it suffices to 
learn a linear classifer of the form $h_w$, ie, one without a bias term. 
And let's assume that the loss function is zero-one loss.
Then, minimising the training error means solving the following optimisation problem:[^logregcode]

```{math}
:label: linzo
\min_{w} L_S(h_w) &= \min_w\frac{1}{|S|}\sum_{(x,y)\in S}\ell_{0-1}(y,\sgn\circ h_w(x)) \notag \\
 & =\min_{w}\frac{1}{|S|}\sum_{(x,y)\in S}\tfrac{1}{2}(1-y\sgn\trsp wx)
```

[^logregcode]: {-} [{{ codeicon }}logreg](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/logreg.ipynb)


(Recall that $\ell_{0-1}$ denotes the zero-one loss.)
Unfortunately, this optimisation problem is a hard problem and also in practice not easily solvable. 
Why is it hard? It is not even smooth: A small change in $w$ may flip $\sgn \trsp wx$ from $+1$ to $-1$.

A common learner for linear predictors is *logistic regression*.[^lrname] 
Logistic regression does not try to solve {eq}`linzo` directly but tries to solve 
a *surrogate* problem that is much easier.
This surrogate problem is based on the *logistic function*[^logistic]

$$
\sigm:\mathbb R\to\mathbb R_+,\quad z\mapsto \frac{1}{1+e^{-z}},
$$

which can also be seen in {numref}`logisfig`. 

[^lrname]: {-} Everything is wrong about the name *logistic regression*. It doesn't have anything to 
do with logistics and it's not regression, it's classification.

[^logistic]: By the way, as one can learn on [wikipedia](https://en.wikipedia.org/wiki/Logistic_function) the logistic function has no relation to logistics at all. Rather, the first author to describe the function thought it resembled the *logarithmic* function.

```{figure} pix/logistic.png
:name: logisfig
:width: 15cm

The logistic function
```

The optimisation problem that logistic regression solves is:

```{math}
:label: logloss
\min_{w} \frac{1}{|S|}\sum_{(x,y)\in S}-\log_2\left(\sigm(y\trsp wx)\right)
```

Why is the surrogate problem easier? 
Because it is a smooth optimisation problem and thus can be solved with a number
of numerical optimisation algorithms. In fact, it is even a *convex optimisation* problem, 
a particularly simple type of optimisation problem that we'll
discuss later in this course. For the moment 
let us simply observe that the zero-one loss in {eq}`linzo` is upper-bounded by 
the so called *logistic loss* in {eq}`logloss`:

```{prf:Lemma}
:label: upperloglosslem
For all training sets $S\subset\mathbb R^d\times\{-1,1\}$ and 
$w\in\mathbb R^d$ it holds that 

$$
\frac{1}{|S|}\sum_{(x,y)\in S}\ell_{0-1}(y,\sgn\circ h_w(x))
\leq \frac{1}{|S|}\sum_{(x,y)\in S}-\log_2\left(\sigm(y\trsp wx)\right)
$$
```

```{prf:Proof}
We show that for all $(x,y)\in S$

$$
\ell_{0-1}(y,\sgn\circ h_w(x)) \leq -\log\left(\sigm(y\trsp wx)\right).
$$

First, assume that the zero-one loss of $(x,y)$ with respect to $h_w$ is 0. 
Then  $y\trsp wx\geq 0$ and

$$
-\log\left(\sigm(y\trsp wx)\right) = \log\left(1+e^{-y\trsp wx}\right) \geq \log(1) \geq 0=\ell_{0-1}(y,\sgn\circ h_w(x))
$$

Next, assume the zero-one loss is 1. Then $y\trsp wx\leq 0$ and $\sigm(y\trsp wx)\leq \tfrac{1}{2}$, 
which implies

$$
-\log_2\left(\sigm(y\trsp wx)\right) \geq -\log_2(\tfrac{1}{2}) = 1 = \ell_{0-1}(y,\sgn\circ h_w(x))
$$
```

As a consequence of the lemma, a decrease in the logistic loss will tend to decrease
the the training error 
as well. 


Training and errors
-------------------

So far, we have tried to minimise the training error. With the zero-one-loss, for instance,
we tried to minimise the misclassification rate on the training set when fitting a logistic regression.[^ercode]

[^ercode]: {-} [{{ codeicon }}errors](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/errors.ipynb)

```{figure} pix/testerr.png
:name: testerrfig
:width: 15cm

Training, validation and test set.
```


The real question, however, is how well the algorithm performs on *new* data, on data it has not seen
during training. 
A classifier is worthless if it fails on real, new data, however well it classifies the training set. 

Analysing the performance on the training set is easy, simply compute the training error --- 
but how can we evaluate the real-life performance? 

In practice, we split our data into two parts, a training set and a *test set*. Typically, 
the training set is larger, perhaps 80\% of the available data, because the algorithms learn better 
with more data. As before, the learning algorithm has access to the training set during the training 
phase. *It must not have access, in any way, to the test set.*
It's best to imagine that the test set is locked away in a high security
vault and only retrieved when the classifier is completely learnt. 

In fact, the learning phase typically involves a bit (or rather, a lot) of fiddling. Most machine 
learning algorithms offer a number of knobs and buttons. To find the 
best set of settings the algorithms are trained with different settings (on the training set) 
and then evaluated, ie, their training error computed. (Or, if there's data to spare, a sort of 
test set of the training set, sometimes called *validation set*, is used for evaluation.)

Only when the algorithm is finished, we apply the classifier to the test set in order to compute
the *test error*. More formally, if $\mathcal X$ is the domain set, $\mathcal Y$ the set of classes, 
$h$ the classifier and $\mathcal T\subseteq \mathcal X\times\mathcal Y$ the test set, 
then the test error is

$$
\frac{1}{|\mathcal T|}\sum_{(x,y)\in\mathcal T}\ell(y,h(x))
$$

The idea here is that, as the algorithm has never seen the test data, the test error approximates
the error the classifier will make on real-life data. 
We will later also obtain some theoretical guarantees for the performance on new data.

In practice the test error is always larger than the training error, and sometimes substantially larger.
The aim therefore is to obtain first a relatively small training error, and then a small gap
between training and test error.

(polysec)=
Polynomial predictors
---------------------

Linear predictors are very simple and will often have a large training error.[^quadcode] 
A classifier based on quadratic functions, or even higher polynomials, 
will certainly be more powerful and will often have smaller training error.
For a quadratic classifier, 
we look for weights $u\in\mathbb R^{d\times d}$, $w\in\mathbb R^d$ and $b\in\mathbb R$
such that 

$$
y=1 \text{ if and only if }\sum_{i,j=1}^nu_{ij}x_ix_j+\sum_{i=1}^dw_ix_i+b\geq 0 \text{ for every }(x,y)\in S
$$

Fortunately, this simply reduces to a linear predictor -- if we are willing to modify our training set. 
Indeed, transform each $x\in\mathbb R^d$ into a vector $\overline x\in\mathbb R^{d^2+d+1}$ by 
putting

$$
\overline x=\trsp{(x_1x_1,x_1x_2,\ldots,x_1x_d,x_2x_1,x_2x_2,\ldots, x_dx_d,x_1,x_2,\ldots,x_d,1)}
$$

Then, with the new training set

$$
\overline S=\{(\overline x,y):(x,y)\in S\}
$$

a linear predictor is the same as a quadratic predictor on $S$. Obviously, we can do the same for 
higher polynomials.

[^quadcode]: {-} [{{ codeicon }}quadpred](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/quadpred.ipynb)

```{figure} pix/quadpred.png
:name: quadpredfig
:width: 8cm

A quadratic classifier
```

Nearest neighbour
-----------------

A classic and very simple algorithm is *nearest neighbour*.[^nearcode] 
During training
the algorithm simply memorises all training data points. When it is tasked to predict the 
class of a new data point it determines the closest training data point and outputs the class
of the training data point. The idea is that two data points that have similar features are 
likely to have the same class.\movtip{neighbour}% 

As described the algorithm is very sensitive towards noise in the training set. 
A single erroneously classified data point in the training set, for instance, may lead to 
many bad predictions of new data. Because of that, a more robust variant is often used, 
*$k$-nearest neighbour*: for each new data point the $k$ closest data points of the 
training sets are determined, and the output is then most common class among these
$k$ data points. (Ties may be split randomly.)

[^nearcode]: {-} [{{ codeicon }}nearest](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/nearest.ipynb)

```{prf:Algorithm} $k$-nearest neighbour
:label: nnalgo

**Instance** A training set $S$, a new data point $x$.\
**Output** A predicted class for $x$.

1. Set Determine the $k$ closest points $s_1,\ldots,s_k$ in $S$ to $x$.
2. **output** the majority class of $s_1,\ldots, s_k$. 
```

Some details in the algorithm are still vague. What, for instance, does "closest" mean? 
Here, several options are available and ultimately should be considered in context of the 
application. In the simplest case, we take the euclidean distance. Some preprocessing might be 
necessary, though, if the features have very different scales. Consider a data set 
of customers, where the features include age and yearly income. The age difference between
two customers should, if we are generous, be at most a 100 years. The difference between
two yearly incomes may easily surpass 10000€, and a yearly income difference of a 100€ is
 trivial. That is, if we do not rescale these two features such that they have similar
magnitudes we will weigh an income difference of a 100€ as much more serious as an
age difference of 50 years. A 15 year old, however, will have  interests that are quite different
from the ones of a 65 year old person. 
However, there is very little difference between
 a yearly income of 45000€ and an income of 45100€.


````{subfigure} AB
:name: knnfig
:layout-sm: A|B
:gap: 10px

```{image} pix/one_nn.png
:alt: one
:width: 6cm
```

```{image} pix/twenty_nn.png
:alt: twenty
:width: 6cm
```

Algorithm $k$-nearest neighbour for $k=1$ on the left and $k=20$ on the right
````

Feature scaling is not only a problem for $k$-nearest neighbour but also for many other 
machine learning algorithms. 


Next, how do we find the $k$-closest points in the training set? If the 
training set is large then going through all points in the training set in order to compute
the distance to $x$  will be computationally expensive. 
There are some tricks that may speed up this step considerably. We may discuss these later. 

Nearest neighbour has the advantage that it is dead-simple. It's also relatively easy to 
analyse from a theoretical point of view. In practice, it suffers from a number of 
drawbacks. The most obvious is that it is quite memory intensive: it has to store the 
whole training set!

(decsec)=
Decision trees
--------------

Let $\mathcal X\subseteq \mathbb R^d$, and let $\mathcal Y$ be a (finite) set of classes.[^deccode]
A \defi{decision tree} $T$ for $\mathcal X,\mathcal Y$ consists of a rooted tree, where each 
non-leaf is associated with a decision rule. That is, $T$ has a root $r$, every vertex $v$ is either
a \ndefi{leaf} or not; if $v$ is not a leaf then it has precisely two children $v_L,v_R$; every
leaf $v$ is labelled with a class $c(v)\in\mathcal Y$; every non-leaf $v$ has a decision rule, 
a tuple $(i,t)$, where $i\in[d]$ is a feature and $t\in\mathbb R$ 
is a threshold. A decision tree defines a classifier in the following sense:

[^deccode]: {-} [{{ codeicon }}dectree](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/dectree.ipynb)

```{prf:Algorithm} decision tree
:label: dectreealgo

**Instance** $x\in\mathcal X$.\
**Output** a class $y\in \mathcal Y$.

1. Set $v=r$, where $r$ is the root of the tree.
2. **while** $v$ is not a leaf:
3.   {{tab}}Let $(i,t)$ be the decision rule of $v$.
4.   {{tab}}If $x_i\leq t$ set $v=v_L$; else set $v=v_R$.
7. **output** the class $c(v)$ of $v$.
```

```{figure} pix/iris.png
:name: iristreefig
:width: 15cm

A decision tree for the *iris* data set
```

$\newcommand{\gain}{\text{Gain}}$

How is a decision tree computed? The tree is constructed iteratively, starting with the root. 
Each vertex $v$ is associated with a subset $S_v$ of the training set $S$, namely with those that end up 
in the vertex if we follow the already existing decision rules. Now, if all data points in $S_v$
have the same class $y$, $v$ is declared  a leaf and its class is fixed to $c(v)$.
If not, $v$ receives two children $v_L,v_R$ and a decision rule $(i,t)$ -- the decision rule 
is picked in order to maximise a measure, $\gain_{S_v}(i,t)$ that depends on the feature $i$ and 
on a threshold $t$. 
The idea is that $\gain_{S_v}(i,t)$ is largest if the decision rule $(i,t)$ 
gives the *best* split (whatever that 
may mean). We discuss $\gain$ below.

```{prf:Algorithm} tree growing
:label: growalgo

**Instance** Training set $S$\
**Output** A decision tree

1. **function** Tree($S$)
2. {{tab}}Let $v$ be a new vertex. 
2. {{tab}}If all points in $S$ have the same class $y$, set $c(v)=y$ and return leaf $v$.
2. {{tab}}Let $(i^*,t^*)$ be the decision rule that maximises $\max_{i\in[d],t\in\mathbb R}\gain_S(i,t)$
2. {{tab}}Associate rule $(i^*,t^*)$ with $v$.
2. {{tab}}Set $S_L=\{(x,y)\in S: x_{i^*}\leq t^*\}$ and $S_R=S\setminus S_L$.
2. {{tab}}Let $T_L$ be the tree returned by Tree($S_L$) and $T_R$ be the one returned by Tree($S_R$).
2. {{tab}}Let $v_L$ be the root of $T_L$ and $v_R$ the root of $T_R$.
2. {{tab}}Set $v_L,v_R$ to be children of $v$ and return the resulting tree. 
7. **output** Tree($S$).
```


The algorithm is a basic model to construct a decision tree -- there are many variants. In particular, 
actual implementations normally provide 
a number of  ways to limit the growth of the tree. Often, one may set a maximal depth (largest length of a root-leaf path) or a 
minimal number of data points for leaves. 



% discussion of gain measures:
% https://sebastianraschka.com/faq/docs/decision-tree-binary.html
What about the gain measure?[^Rasch]  
In general, the gain depends on an impurity measure $\gamma(S')$ of a subset of $S$. 
The impurity $\gamma(S')\in [0,1]$ measures, in some appropriate way, how heterogeneous the set $S'$ is. 
Thus, we should have $\gamma(S')=0$ if all data points in $S'$ are of the same class; and a larger value
if $S'$ is a mixture of several classes. There are a number of different impurity measures that 
we will discuss below.

[^Rasch]: Partially based on Sebastian Raschka's [Machine Learning FAQ](https://sebastianraschka.com/faq/docs/decision-tree-binary.html)

The gain
is computed as follows:

$$
\gain_{S_v}(i,t)=\gamma(S_v)-\left(\frac{|S_L|}{|S_v|}\gamma(S_L)+\frac{|S_R|}{|S_v|}\gamma(S_R)\right)
$$

Here, $S_L$ and $S_R$ is the split of $S_v$ according to the decision rule. That is

$$
S_L=\{(x,y)\in S_v: x_i\leq t\}\text{ and }S_R=S_v\setminus S_L
$$

How the impurity measure $\gamma$ is defined, differentiates the different spliting criterions.
Note that maximising \gain\ amounts to \emph{minimising} 

```{math}
:label: impurity
\left(\frac{|S_L|}{|S_v|}\gamma(S_L)+\frac{|S_R|}{|S_v|}\gamma(S_R)\right),
```
which can be seen as a measure of how heterogenous the children of the node $v$ are. 

For $S'\subseteq S$ 
and $y'\in \mathcal Y$ put 

$$
p(y',S')=\frac{\# \{(x,y)\in S': y=y'\}}{|S'|}
$$

(This is the fraction of points that have label $y'$, and may be interpreted as the probability
that a randomly picked point in $S'$ has label $y'$.)

To minimise training error in each step, $\gamma$ is set to 

$$
\gamma(S')=1-\max_{y\in\mathcal Y}p(y,S')
$$

To minimise the *Gini impurity*,[^gini] set $\gamma$ to 

$$
\gamma(S')=1-\sum_{y\in\mathcal Y}p(y,S')^2
$$

There is also a third common measure, where $\gamma$ becomes the *entropy* of $S'$

$$
\gamma(S')=-\sum_{y\in\mathcal Y}p(y,S')\log_2 p(y,S')
$$

Often, however, there is not much of a difference between $\gain$ based on
Gini impurity and based on entropy. 
 
[^gini]: {-} Gini impurity should not be confused with 
the Gini coefficient; the latter measures the homogeneity of a random variable, such 
as income in a country.
 
In any case, we observe that $\gamma(S')$ becomes 0 when all points in $S'$ belong to the same 
class. {numref}`ginifig` shows $\gamma(S')$ for two classes and different values of $p(y,S')$
of the first class.  

```{figure} pix/gini.png
:name: ginifig
:width: 12cm

Train error, Gini impurity and entropy for two classes; $x$-axis shows fraction of first class in sample.
```

Decision trees have a number of advantages. First, they are able to handle any number of classes. That is, 
in contrast to linear classifiers they are not restricted to binary classification. Second, they 
are rule-based, which makes them easy for us, the human users, to understand. Decision trees are not
black-boxes (this point, though, is less true, the larger the tree).
Third, as each decision rule operates on a single feature, there is no need to worry about scaling: 
features of wildly different scales (ages and yearly incomes, say) do not pose a problem. 
Finally, decision trees can also easily handle missing data.
Nevertheless, I don't think that decision trees are still widely used -- their performance is simply
not good enough. If many decision trees are combined, however, we obtain a simple but quite accurate
predictor, a *random forest*. We'll discuss random forests later.  


Neural networks
---------------

We've already talked about a number of classifying algorithms. I omitted, however, the most important one of 
them all, the neural network. Let's remedy this omission.[^tfcode]

[^tfcode]: {-} [{{ codeicon }}tfintro](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/neural_networks/tfintro.ipynb)

Neural networks are old: Their origins go back to the 1940s. Modelled after (a crude simplification of) neurons
in the brain they consists of single units, the neurons, that collect the inputs of other units (or of the input)
and then compute an output value that might be fed into other neurons. 

```{figure} pix/neuron.png
:name: neuronfig
:width: 6cm

A single neuron
```

Mathematically, a neuron is simply a function $f:\mathbb R^n\to\mathbb R$ of a special form. 
Namely, it's the concatenation of an affine function $x\mapsto Wx+b$ with a (usually) non-linear
 function $\sigma:\mathbb R\to\mathbb R$, the *activation* function:

$$
f(x) = \sigma(Wx+b)
$$
 
The rows of the matrix $W$ are often called the *weights* of the connection between the
input and the neuron, while the number $b$ is the *bias*. Common activation functions
are shown in {numref}`actfig`. While historically the logistic function and tanh 
were widely used, nowadays the *rectified linear unit* (short ReLU)
is much more common. It is defined as $z\mapsto \max(0,z)$. A variant, *leaky ReLU*
takes a (fixed) parameter $\alpha$, typically a small but positive value,
and then is given by $z\mapsto \max(\alpha z,z)$.

```{figure} pix/activations.png
:name: actfig
:width: 15cm

Common activation functions
```

A single neuron obviously does not constitute a network. In a neural network neurons 
are organised in *layers* that feed their output into the following layer. In the 
traditional design every neuron of some layer forwards its input to every neuron 
of the following layer. We then talk about a *fully connected* layer. Activation 
functions are the same for all neurons within a layer but may vary between layers.
This is in particular so for the *output layer*, the last layer. Here, we often 
use the sign function $\sgn$ (well, quite rarely in practice) and even more often 
the logistic function, or the softmax function (which we'll discuss later). 
Why the logistic function? Because its range of $[0,1]$ 
can quite conveniently be interpreted as a probability, or as a level of confidence. 
That is, we see the output of $0.74$ as a confidence of $74\%$ that the picture
shows indeed a cat. 


More formally, we can describe a fully connected neural network with $n$ inputs, $L-1$ hidden layers and
one output (i.e.\ a network that could be used for binary classification)
 as follows. 
Each layer $\ell$ 
has a certain width $n_\ell$, the number of nodes in that layer, where we put $n_0=n$.
Between layer $\ell$ and $\ell-1$ (where layer 0 is the input layer, and where layer $L$
is the output layer) there are weights, represented by the matrix $W^{(\ell)}\in\mathbb R^{n_\ell\times n_{\ell-1}}$.
Each neuron $h$ in the $\ell$th layer has a bias $b^{(\ell)}_h$, the vector of biases of the $\ell$th layer 
is $b^{(\ell)}\in\mathbb R^{n_\ell}$. 
Let's assume that each layer has the same activation function $\sigma:\mathbb R\to\mathbb R$ (most commonly ReLU), 
except 
for the output layer. 


Set $f^{(0)}(x)=x$.
For $\ell=1,\ldots, L$ we compute the input of the $\ell$th layer as

$$
g^{(\ell)}(x)=W^{(\ell)}f^{(\ell-1)}(x)+b^{(\ell)}
$$

and for $\ell=1,\ldots, L-1$ the output as 

$$
f^{(\ell)}(x)=\sigma(g^{(\ell)}(x)),
$$

where $\sigma$ is applied componentwise.
The output of the network is then $F(x)=\sgn(g^{(L)}(x))$ or $F(x)=\sigm(g^{(L)}(x))$,
depending on whether we want just the class or also a confidence level for the class. 
We also compute the number of parameters of the network 
as $N=\sum_{\ell=1}^{L} n_{\ell}\cdot n_{\ell-1} + n_{\ell}$.

```{figure} pix/classnet.png
:name: classnetfig
:width: 15cm

A simple classification ReLU-neural network
```

How now is the network used for classification? Simple, with $\sgn$ as output layer activation
it realises a function $F:\mathbb R^n\to \{0,1\}$, which can directly be used for classification.
More interesting is the question: How do we train a neural network? That is, how do we find 
the weights and biases? We'll talk about this later. 

Modern, high performance neural networks, of the sort that power Alexa, Siri and however the Google Siri
should be called,
have many layers, many neurons and more complicated topologies than the simple fully connected network presented
here. They also have a larger output layer: For a classification task with $k$ layers the network usually
has an output layer of size $k$, with softmax activation, which we'll discuss later, too.

````{dropdown} Neural networks keep on growing
:color: success
:icon: telescope

Neural networks are becoming ever bigger. 

1989
: LeCun's neural network for handwritten digit recognition had about 66000 parameters in three layers;

2012
: AlexNet won the ImageNet competition, with 62 million parameters in 7 layers;

2015
: The winner of the 2015 ImageNet competition had 152 layers and about 60 million parameters.

2020 
: GPT-3, a generative natural language processing network (meaning: it talks), has 175 billion parameters.

The human brain, in contrast, is said to have 10{sup}`15` connections, which should probably be compared
to the number of parameters of a neural network. In 2024 the wiring diagram of the complete brain of a fruit fly was 
finished. It turns out that a fruit fly has about 140000 neurons and 54.5 million synapses.

[tensorflowcv](https://pypi.org/project/tensorflowcv/)<br/>
[Wikipedia entry on GPT-3](https://en.wikipedia.org/wiki/GPT-3)<br/>
*How to map the brain*, S. DeWeerdt, Nature (2019)<br/>
*Largest brain map ever reveals fruit fly’s neurons in exquisite detail*, [Nature](https://www.nature.com/articles/d41586-024-03190-y) (2024)
````

Loss functions
--------------

So far, we have only considered zero-one loss. 
In some situations, other loss functions will be needed. For instance, if we want to classify
emails as *spam or ham*, that is, as spam emails or non-spam, then it is far more serious to 
misclassify a ham email than a spam email: Indeed, if we miss an important email because it was put 
in the spam folder (or perhaps deleted) then we will be quite cross, the occasional Viagra ad in our 
inbox, however, will merely annoy us somewhat. 
A loss function then might look like this:

$$
\ell(y,y')=
\begin{cases}
0 & \text{if }y=y'\\
1 & \text{if }y=\textsf{ham}\text{ but }y'=\textsf{spam}\\
0.1 & \text{if }y=\textsf{spam} \text{ but } y'=\textsf{ham}
\end{cases}
$$

Zero-one loss is symmetric in its two arguments; the loss function here isn't. Let us agree on 
the convention that the first argument is the true value, while the second one is the 
predicted value. That is, for a classifier $h$ we would consider $\ell(y,h(x))$
when computing the training error.


Regression
----------

Regression is a more general problem than classification. 
In both cases, we aim to find a predictor $h:\mathcal X\to\mathcal Y$. In classification, the target set
$\mathcal Y$ is finite: we are interested in knowing whether the email is spam or not, or 
whether the image shows a cat, a dog or a hamster. In regression, in contrast, $\mathcal Y$ is usually
continuous, and normally equal to an interval $[a,b]$, or perhaps to a multidimensional analog 
such as $[a,b]^n$. 
In principle, regression can be seen as a *function approximation* task. There is some unknown function $f$
that we want to approximate with our predictor $h:\mathcal X\to\mathcal Y$.[^regcode]

[^regcode]: {-} [{{ codeicon }}regression](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/california.ipynb)

The theory for classification, however, is cleaner than for regression, and this is the main reason 
why I will focus on classification tasks. 
Arguably, regression is more powerful.
I am probably more interested in how large the expected return on investment
for some stock portfolio is (a regression task) than whether there is a positive or negative ROI
(a classification task). Let's do a quick digression on regression.

Fortunately, most classification algorithms can be repurposed for regression. 
*Linear regression*
is probably best known. Given a training set $S\subseteq\mathbb R^d\times\mathbb R$ find 
a vector $w^*\in\mathbb R^d$ and a bias $b^*$ such that 

$$
\min_{w,b} \sum_{(x,y)\in S} ||\trsp wx+b-y||_2^2
$$

is minimised. Here we have used *squared error loss* as a loss function,
which is quite common for regression problems.

To use $k$-nearest neighbour for regression, find for some sample $x$ the $k$ closest
data points in the training set and then output the average of their target values. (Perhaps weighted
by distance to $x$.) Decision trees are adapted in a similar way.

````{dropdown} Examples of regression tasks
:color: success
:icon: telescope

physics modelling
: Many interatomic potentials in physics are described by 
solutions of complex partial differential equations. Neural networks can sometimes approximate the solutions
of these PDEs much faster than other numerical methods.  

house prices
: Real estate portals often offer a quick algorithmic house price estimation
based on a few key data (location, age of the house, size). 

credit scores
: On each credit application the financial data of the applicant is 
evaluated to compute a credit score, a number that expresses the confidence of how likely
the applicant is to pay back the credit. In Germany, the credit score is normally 
provided by SCHUFA. How credit scores are used is sometimes 
morally problematic and sometimes outright evil.

*The rise of data-driven modelling*, Editorial, Nature (2021)<br/>
*Weapons of Math Destruction*, Cathy O'Neil, Chapter 8 (2016)
````

A statistical model
-------------------

On the face of it, what machine learning algorithms accomplish seems implausible. 
We train the predictors on one dataset, the training set, and then expect
the predictors to perform well on a completely different dataset, ie, on new data, or 
on the test set. How can that work at all? 

That works -- if training data and new data come from the same source. Mathematically, 
we model this by a probability distribution  $\mathcal D$ on $\mathcal X\times\mathcal Y$.
That is, we assume that there is distribution that produces pairs $(x,y)\in\mathcal X\times\mathcal Y$
of datapoints $x$ and classes $y$, and we assume that the training set as well as 
any new data (or the test set) is drawn from this distribution. 

Why a probability distribution? Datapoints occur in nature with different frequencies. 
For instance, if we try to recognise hand-written letters 
we expect to see far fewer "z" than "e", and consequently, it would probably be less critical to sometimes 
misclassify a "z" than to sometimes misclassify an "e". We normally assume a *joint* 
distribution on $\mathcal X\times\mathcal Y$ because the class is not always completely determined by the 
features. 

For example, assume we are predicting whether someone will default on their credit, and 
the features we have access to are income, credit history, and employment status. Now, there might be 
two people with exactly the same features, Alice and Bob, but Bob defaults on his loan
but Alice does not. This may be because the data is incomplete (Bob became ill, and poor health is 
perhaps not well predicted by income and so on), or because the data is faulty (Alice misrepresented
her income) or because the outcomes are simply not deterministic.
That is, there is an inherent uncertainty in the data. To capture this, we imagine 
there is an underlying probability distribution that would, if we knew it, that encapsulates information 
such as "someone with this income, that age and this credit history has a risk of 11.374\%
of defaulting on their credit". 
%Obviously, we never know this probability distribution. (If we did
%there would not much be to predict; we'd simply give a credit to everybody with a risk of smaller than 10\%, say.)

What else do we assume?

* The distribution $\mathcal D$ is *unknown* and we normally do not make
any assumptions about it. It could be wild.

* If we draw several datapoints, we draw them in *iid* fashion. That is, 
the data points are *identically and independently distributed*.

* The distribution is *fixed* -- it does not change with time. This is actually an assumption 
that will not be satisfied in many real-world applications. While dogs will (probably) still look like dogs 
in ten years, the features that strongly indicate credit-worthiness now might be much less indicative in ten years.   


What is this model good for? We can formulate our ultimate goal in classification: to learn a 
classifier $h:\mathcal X\to\mathcal Y$ with low average error on *all* data points, including
unknown data points. More formally, 
we want to obtain a small *generalisation error*:

$$
L_{\mathcal D}(h)=\expec_{(x,y)\sim\mathcal D}[\ell(y,h(x)]
$$
 
So, this is the expected loss (remember that $\ell(\cdot,\cdot)$ is a loss function, often the zero-one loss)
over the distribution $\mathcal D$. Other names for the generalisation error are *true risk*
or simply *risk*. Note that for the zero-one loss, we have:

```{math}
:label: zorisk
L_{\mathcal D}(h)=\proba_{(x,y)\sim\mathcal D}[h(x)\neq y]
```
That is, the generalisation error is equal to the probability that the classifier misclassifies a data point.

|
Let me stress that the distribution $\mathcal D$ is the link between training set
and real-life datapoints that makes learning possible. We rely here on a major assumption:

> training samples and new samples are independently drawn, and from the same distribution.

Sometimes this assumption is violated, and the mechanism that generates training samples
differs in some way from real-life samples. In that case, all error guarantees go out of the window.

(sec:bayeserr)=
Bayes error
-----------

Because
 of the inherent uncertainty in the distribution, the smallest generalisation error that can be achieved
might be larger than zero. 

Let's assume, for example, that we want to predict the sex of a person based only on the height of the 
person. A person of height 1.8m is likely male -- but not necessarily so. Obviously, there are also 
women of 1.8m, but they are fewer than men of 1.8m. Height does not determine sex, and thus any
classifier based only on height will never be perfect. This residual error that any classifier 
is bound to make is called *Bayes error*. It is defined as 

$$
\bayerr=\inf_h L_{\mathcal D}(h),
$$

where for technical reasons the infimum is taken over all measureable classifiers $h$.
A classifier that attains the Bayes error is called *Bayes classifier*.

```{figure} pix/heights_notes.png
:name: heightsfig
:width: 8cm

Height distribution of US-Americans in 2007 / 2008 of age 40 to 49, fitted to Gaussians. Source: U.S.\ National Center for Health Statistics
```

Consider {numref}`heightsfig`, where the heights of men and women between 40 and 49 in the US
are tabulated. The bars represent the results of a survey, while the solid lines are fitted Gaussians. 
We take these Gaussians as ground truth. That is, we assume that they indicate the probabilities[^condprob]
$\proba[\text{female}|\text{height}]$ and $\proba[\text{male}|\text{height}]$.
We assume zero-one loss.
Given the probabilities, what is the best way to classify by height? Up to a height of approximately 1.71m the 
probability is larger that the person is female, and we'd predict "female". 
From 1.71m on we'd predict that the person is male. 

[^condprob]: These are *conditional probabilities*; see the Appendix for a formal definition. Briefly, if $\proba$
models a joint probability distribution on $\mathcal X\times\mathcal Y$ then $\proba[y|x]$ is defined as 
$
\proba[y|x]=\frac{\proba[(x,y)]}{\proba[x]},
$
where $\proba[x]=\sum_{y'}\proba[(x,y')]$ is the marginal probability.

The resulting classifier would be:

```{math}
:label: bayclass
\bayes(x)=\argmax_{y\in \mathcal Y}\proba[y|x] 
```

(Note that this definition also works for non-binary classification.) Formally, we still need to 
prove that the classifier is actually a Bayes classifier. We'll do that below. 
Moreover, there is some ambiguity if two classes have the exact same probability. In that case,
let us agree that $\argmax$ simply returns one of the classes with maximal probability.

%For a given
%data point $x\in\mathcal X$, let us denote by $\text{noise}(x)$, the \defi{noise at $x$},
%the probability that $\bayes$ misclassifies $x$. What is that probability? 
%Let $y^*\in\mathcal Y$ be the class returned by $\bayes$, ie, $y^*=\bayes(x)$. 
%Now, the probability that $x$ has  class $y$ is $\proba[y|x]$. If $y=y^*$ 
%the classifier is correct. The remaining probability is 
%\[
%\text{noise}(x)=1-\proba[y^*|x] = 1-\max_{y\in\mathcal Y}\proba[y|x]
%\]
%Now, assuming Proposition~\ref{bayesprop} below we obtain
%\[
%\bayerr=L_{\mathcal D}(\bayes) = \proba_{(x,y)\sim\mathcal D}[\bayes(x)\neq y]
%= \expec_x[\text{noise}(x)]
%\]
%That is, the Bayes error is equal to the average noise. 

Let us prove that $\bayes$ as defined in {eq}`bayclass` is indeed the Bayes classifier.

```{prf:Proposition}
:label: bayesprop

Let $h:\mathcal X\to \mathcal Y$ be a classifier, let $\mathcal D$ be a distribution over $\mathcal X\times\mathcal Y$
with zero-one loss.
Then the generalisation error of $h$ cannot be smaller than the Bayes error:

$$
L_{\mathcal D}(h)\geq L_{\mathcal D}(\bayes), 
$$

where $\bayes$ is defined as in {eq}`bayclass`.
```

% I think this should only be true for zero-one loss!
% That is, Bayes classifier should be defined differently for other loss functions

````{prf:Proof}
To avoid annoying integrals, probability density functions and technicalities, let me prove the statement for 
finite or countable domain sets $\mathcal X$. The proof for general $\mathcal X$ follows along 
the same lines. 

The true risk is now
\begin{align*}
L_{\mathcal D}(h)& =\expec_{(x,y)\sim\mathcal D}[\ell_{0-1}(y,h(x))]\\
& = \sum_{x\in\mathcal X,y\in\mathcal Y,h(x)\neq y}\proba[(x,y)]\\
& = \sum_{x\in\mathcal X}\proba[x]\left(\sum_{y\in\mathcal Y,y\neq h(x)}\proba[y|x]\right)\\
& = \sum_{x\in\mathcal X}\proba[x]\left(1-\proba[h(x)|x]\right)\\
& \geq \sum_{x\in\mathcal X}\proba[x]\left(1-\max_{y\in\mathcal Y}\proba[y|x]\right)\\
& = \sum_{x\in\mathcal X}\proba[x]\left(1-\proba[\bayes(x)|x]\right)\\
& = \expec_{(x,y)\sim\mathcal D}[\ell_{0-1}(y,\bayes(x))] = L_{\mathcal D}(\bayes)
\end{align*}
````


%Note that a similar result is true for other loss functions, as long as the Bayes classifier
%is defined accordingly. That is, in the spam-or-ham example it should tend more to class 
%an email as spam than to class it as ham. 

Let me stress that the Bayes classifier is a purely theoretical 
construct. As we do not know the distribution we cannot compute it!
The Bayes error is likewise not a quantity that we can routinely determine. 
In some situations, however, it may be possible to estimate bounds for the 
Bayes error.[^bayerr]
It is unclear to me how well that works. 

[^bayerr]: See for instance *Evaluating Bayes Error Estimators on
Real-World Datasets with FeeBee*, C. Renggli et al.\ (2021), [arXiv:2108.13034](https://arxiv.org/abs/2108.13034)
} 


While neither the Bayes classifier nor the Bayes error are accessible to us, they are still 
useful concepts. The Bayes error makes clear that in some tasks it may not be possible to 
improve a classifier much further -- simply because its error rate comes close to the Bayes error.




