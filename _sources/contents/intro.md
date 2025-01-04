$\DeclareMathOperator{\sgn}{sgn}
\newcommand{\trsp}[1]{#1^\intercal} % transpose
\newcommand{\sigm}{\phi_{\text{sig}}} % logistic function
\newcommand{\twovec}[2]{\begin{pmatrix}#1\\#2\end{pmatrix}}
$

Predictors, classification and losses
=====================================

Does a given picture show a dog or a cat?
Should the applicant be granted a credit? Is the mail
spam or ham? These are all examples of *classification* tasks, one of the 
principal domains in machine learning. 

A well-known classification tasks involves the MNIST data set.[^mnist]
The MNIST data set contains 70000 handwritten digits as images of $28\times 28$ pixels. The task is 
to decide whether a given image $x$ shows a $0$, $1$, or perhaps a $7$. In machine learning
this tasks is solved by letting an algorithm *learn* how to accomplish the tasks by 
giving it access to a large number of examples, the *training set*,
together with the true classification.[^mnistcode] 
In the MNIST tasks that means that the algorithm not only
receives perhaps 60000 images, each containing a handwritten digit, but also for each image
the information which *class* it is, ie, which of the digits 0,1,...,9 is shown; see {numref}`mnistfig`.  

[^mnist]: {-} MNIST is so well known that it has a [wikipedia](https://en.wikipedia.org/wiki/MNIST_database) entry.
[^mnistcode]: {-} [{{ codeicon }}MNIST](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/basic_algorithms_and_concepts/MNIST.ipynb)

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
:name: gdtimefig
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
For MNIST the tuples $(x,y)$ in the training set would consist of the image~$x$ of a digit and $y\in\{0,\ldots, 9\}$, 
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


