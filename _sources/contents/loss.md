$\newcommand{\trsp}[1]{#1^\intercal} % transpose
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\DeclareMathOperator*{\vari}{\mathbb{V}}   % Probability
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\sigm}{\phi_{\text{sig}}} % logistic function
\newcommand{\softmax}{\textsf{soft}}
\DeclareMathOperator{\sgn}{sgn}
\newcommand{\bayes}{h_{\text{Bayes}}} % the Bayes classifier
\newcommand{\bayerr}{\epsilon_\text{Bayes}} % the Bayes error
$


Loss functions
==============

Loss functions play a major role in classification as well as in regression. 
That role, however, is not the same. Let us first examine how loss functions 
determine the quality of a classifier. At the end of this section, we
consider different loss functions for regression.


Loss functions in classification
--------------------------------

A key metric of interest in classification is the misclassification rate, ie, 
zero-one loss. Directly minimising zero-one loss, however, is usually 
computationally infeasible. Instead, we minimise *surrogate* losses, losses
that are better behaved. What are common losses?

In a *binary classification* problem, with classes -1 and 1, we often compute 
a classifier of the type $h:x\in\mathcal X\mapsto \sgn\circ f(x)$, where
$f:\mathcal X\to\mathbb R$ is some function. That is, if $f(x)<0$ then 
we classify $x$ as class -1, and if $f(x)\geq 0$ then $x$ is predicted
to have class 1. In a neural network, or in logistic regression, 
$f$ is passed further through the logistic function, so that the classifier
ultimately computes a probability estimate in $[0,1]$, how likely it is that the sample is class 1.

In this setting a number of loss functions $\ell$ are common. Examples are: 

* *logistic loss* as used in logistic regression:

    $$
    \ell(y,f(x)) = -\log\left(\frac{1}{1+e^{-yf(x)}}\right)
    $$

* *square loss*:

    $$
    \ell(y,f(x))=(y-f(x))^2
    $$

* *exponential loss* as used in AdaBoost, a learning algorithm 
based on an ensemble of decision trees:

    $$
    \ell(y,f(x))= e^{-yf(x)}
    $$


Cross-entropy loss appears to be missing in this list. Cross-entropy loss, in a neural network, is normally used in conjunction with a logistic
layer. That is, the neural network computes $f:\mathcal X\to\mathbb R$, passes the result through logistic activation 
and then applies cross-entropy loss. For a sample $(x,y)$ this results in

$$
x\mapsto -\log\left(\frac{1}{1+e^{-yf(x)}}\right),
$$

which is nothing else then logistic loss. In two of theses losses, we see that the true class $y$ and $f(x)$, the output of the predictor, 
appear as a product $yf(x)$. This is in fact also the case for square loss. Indeed

$$
\ell(y,f(x))=(y-f(x))^2 = \begin{cases}
(1-f(x))^2 = (1-yf(x))^2& \text{ if }y=1\\
(-1-f(x))^2 = (1-yf(x))^2& \text{ if }y=-1
\end{cases}
$$


We can also rewrite zero-one loss in this way. For this, we adapt zero-one loss a bit, so that it also accepts abitrary inputs $f(x)$, 
with the understanding that the ultimate prediction is $\sgn\circ f(x)$:
```{math}
:label: zeroonephi
\begin{align}
\ell_{0-1}(y,f(x)) & =\begin{cases}
0 &\text{ if }\sgn f(x)=y\Leftrightarrow\sgn yf(x)=1\\
1 &\text{ if }\sgn f(x)\neq y\Leftrightarrow\sgn yf(x)=-1
\end{cases} \notag \\
& = 1_{\sgn yf(x)=-1}, 
\end{align}
```
where $1_A$ is the characteristic function of a set $A$, ie, the function that equals 1 if the argument is in $A$ 
and 0 otherwise.

```{figure} pix/loss_functions.png
:name: lossfunsfig
:width: 12cm

Some loss functions used in classification. Shown is $\phi:\mathbb R\to \mathbb R_+$, if the 
loss written as $\ell(y,f(x)) = \phi(yf(x))$.
```


[^binwarn]
Because many loss functions only involve the product $yf(x)$, we focus on loss functions of the form

$$
\ell(y,f(x)) = \phi(yf(x)),
$$

for some function $\phi:\mathbb R\to\mathbb R_+$. 

[^binwarn]: {-} Again, this assumes binary classification.


Bayes consistent loss functions
-------------------------------

What properties should a good loss function satisfy? Recall that logistic, exponential and square loss are *surrogate* loss
functions: We are not really interested in a small logistic loss or a small square loss -- rather, our aim 
is to minimise true risk, ie, expected zero-one loss. Zero-one loss, however, is not smooth and because of that difficult to minimise directly. 
The surrogate losses discussed in the previous section are all smooth and thus easier to minimise. 

As can be seen in {numref}`lossfunsfig`, each of logistic, square and exponential loss upper-bounds zero-one loss. 
That is good, because it means that when the surrogate loss becomes smaller then, usually, zero-loss will decrease as well. 

Is that really enough, though? An admittedly contrived loss function
that assigns a loss of 42 whatever the true class, and whatever we predict, is also an upper bound on zero-one loss, 
but obviously does not help in finding a classifier with small true risk.
Why is a loss function such as logistic loss better? Let's have a look.

We assume that all classifiers $h:\mathcal X\to\{-1,1\}$ are based on functions $f:\mathcal X\to\mathbb R$, 
in the sense that they simply output the sign of $f$:

$$
h(x)=\sgn(f(x))\text{ for all }x\in\mathcal X
$$

Assuming that the data is generated by hidden distribution $\mathcal D$ on $\mathcal X\times\{-1,1\}$, 
we write the true risk as a functional of $f$, where we drop the reference to $\mathcal D$ to simplify notation:

$$
L(f):=L_\mathcal D(\sgn\circ f) = \expec_{(x,y)\sim\mathcal D}[\ell_{0-1}(y,\sgn(f(x)))]
$$

The smallest possible true risk is the *Bayes error*, which is achieved by a *Bayes classifier*:

$$
\bayerr=\inf_g L(g)
$$


Given a loss function $\ell$ defined by a function $\phi:\mathbb R\to\mathbb R_+$ by $\ell(y,f(x))=\phi(yf(x))$,
we define in a similar way the expected loss

$$
L_\phi(f):=\expec_{(x,y)\sim \mathcal D}[\phi(yf(x))]
$$

and also the smallest achievable loss

$$
\epsilon_\phi=\inf_g L_\phi(g)
$$

(Should you worry whether these infimums are well-defined: they are understood to range over all measurable functions $g:\mathcal X\to\mathbb R$.)

What now would be an undesirable outcome of learning with a surrogate loss function $\phi$? 
That we find a $f^*$ that minimises surrogate loss, ie, $L_\phi(f^*)=\epsilon_\phi$ but not 
true risk: $L(f^*)>\bayerr$. Then, $f^*$ would not be a Bayes classifier.

Let us call a loss function $\phi$ Bayes-consistent if that never happens. More formally, 
$\phi:\mathbb R\to\mathbb R_+$ is *Bayes-consistent* if for every sequence $f_1,f_2,\ldots$
of functions $f_i:\mathcal X\to\mathbb R$ with 

$$
L_\phi(f_i)\to \epsilon_\phi\text{ for }i\to\infty
$$

it follows that also 

$$
L(f_i)\to \bayerr\text{ for }i\to\infty
$$


Which loss functions are Bayes-consistent? It turns out that already rather mild conditions 
guarantee Bayes-consistency:

```{prf:Theorem} Bartlett, Jordan and MacAuliffe
:label: consistencythm
Let $\phi:\mathbb R\to \mathbb R_+$ be convex, continuous and differentiable at 0 with $\phi'(0)<0$.
Then $\phi$ is Bayes-consistent.
```
The theorem is a special case of a result by Bartlett et al.[^BJM03]
that gives a full characterisation
of Bayes-consistent loss function that also works for non-convex functions.

[^BJM03]: *Convexity, Classification, and Risk Bounds*, P.L. Bartlett, M.I. Jordan and J.D. MacAuliffe (2003)

With the theorem
it is easy to check that  logistic, square and exponential loss are all Bayes-consistent.

How valuable is the theorem? On the one hand, it does seem very valuable: The conditions on $\phi$
are quite natural, only convexity is debatable -- but in that case we could turn to the 
full version of Bartlett et al.\ of the theorem. Moreover, the theorem provides reassurance that all commonly used
loss functions allow, in principle, to find the best classifier possible, a Bayes classifier.

On the other hand, we have to take into account that Bayes-consistency is really the *bare minimum*
we could demand of a loss function. 
Bayes-consistency only tells us that a sequence $f_1,f_2,\ldots$ of classifiers with 
a surrogate loss $L_\phi(f_i)$ that converges to the smallest loss will *eventually*
converge towards a Bayes classifier. We do not get any information, however, how fast that happens.
It could even be the case that in some step surrogate loss decreases, $L_\phi(f_{i+1})<L_\phi(f_i)$,
but true risk increases, ie, $L(f_{i+1})>L(f_i)$. There are some results, though, that relate
surrogate loss to true risk; see, eg, Zhang (2004).[^Zha04]

[^Zha04]: *Statistical behavior and consistency of classification methods based on convex risk minimization*, T. Zhang (2004)

On top of that, the theorem is, in some sense, too strong. It does not help us 
 to distinguish between different loss functions as all
reasonable loss functions, and some unreasonable ones, turn out to be Bayes-consistent.

% EXERCISES: show that hinge loss and truncated square loss z:\mapsto (\max(1-z,0))^2 are Bayes-consistent

:::{seealso}
[Proof of consistency theorem](consproofsec)
:::
% I have no idea why there is no tooltip

Common loss functions in regression
-----------------------------------

In classification we ultimately aim to minimise the expected number of misclassified samples. 
Because it is not computationally feasible to 
directly minimise zero-one loss, we use surrogate loss functions.

In regression, we normally do not need to deal with computationally infeasible loss functions. 
Often we minimise directly the metric that we are interested in. In contrast, however,
it is not obvious anymore what the key metric is. Depending on the data and on the task at 
hand, different metrics may be appropriate to measure the quality of the regressor. 

Here is a list of commonly used metrics, or loss functions, in regression. Each time we assume that $y^*$
is the true value, while $y$ is the prediction of the regressor. During training, we 
take the mean of all the losses over the training set -- that is the reason why 
we normally talk about *mean* square error, or *mean* absolute error. 
Since that is so common, I list the metrics below with *mean* in the name, even though 
the formulas only concern a single sample.

```{figure} pix/regression_losses.png
:name: reglossfunsfig
:width: 15cm

Square error, absolute error and Huber loss, shown as functions
of the prediction error $y^*-y$.
```


* *mean square error* (MSE)

    $$
    (y^*-y)^2
    $$

    The MSE is probably the most popular metric. If you do not have a good reason 
    for another loss, use MSE.

* *mean absolute error* (MAE)

    $$
    |y^*-y|
    $$

    One good reason to avoid MSE: Large errors will easily dominate the MSE.
    This may happen 
    if the training data is noisy, or may contain large outliers. In that case
    it may be better to use mean absolute error. 

* *Huber loss* 

    $$
    %\text{Huber}(y^*,y)=
    \begin{cases}
    \tfrac{1}{2}(y^*-y)^2 & \text{ for }|y^*-y|\leq\delta\\
    \delta|y^*-y| & \text{ for }|y^*-y|\geq\delta
    \end{cases}
    $$

    where $\delta>0$ is a parameter that the user chooses. Huber loss
    combines MSE and MAE. In the interval $[-\delta,\delta]$ it is MSE, outside
    it is MAE. That means, Huber loss is as robust as MAE against outliers but still
    shares the advantages of MSE. What are these? Often it is more important to 
    reduce an error from $0.9$ to $0.8$ than reduce an error of $0.2$ to $0.1$.
    For MAE both of these are the same, while MSE has a preference for the former.

* *mean absolute percentage error* (MAPE)

    $$
    \frac{|y^*-y|}{y^*}
    $$
     
    First observation: MAPE only works for positive true values $y^*$.
    (MAPE could be extended to all non-zero $y^*$ by taking the absolute value $|y^*|$
    in the denominator. It seems doubtful, though, whether MAPE is the right metric in such a setting.)

    When is MAPE appropriate? Suppose we aim to predict the profit of different investments. 
    If the true profit of an investment is €1 million then a prediction error of €10000
    is probably not serious; if, on the other hand, the true profit is €20000 then a prediction error of €10000
    is a serious issue. MSE, and MAE and Huber loss, would weight the prediction error the same in both 
    cases, MAPE would yield an error of $\tfrac{1}{100}$ in the first case, and an error of $\tfrac{1}{2}$
    in the second case. In short, MAPE yields relative errors, and is perhaps appropriate when 
    the true values are not all of the same order magnitude. 

    A warning: If some true values are very small then predictions for these values may
    dominate MAPE, and indeed, result in an astronomical value for MAPE.  
    
* *mean squared log error*

    $$
    \left(\log(y^*+1)-\log(y+1)\right)^2
    $$

    Mean squared log error may only be used for non-negative values. (The +1 is in there, so that the application 
    of $\log$ always yields a non-negative result.) Looking closely we see that this is just MSE for log-scaled
    values. When is that appropriate? When the values span several orders of magnitude. 

    Assume, for instance, we want to predict the GDP of different countries. Now, in 2021 the GDP
    of China was \$17.7 trillion, while the GDP of Andorra was \$3.3 billion.
    If we use MSE then it basically does not matter what we predict for Andorra (\$10 billion, perhaps?)
    as the MSE will be dominated by the prediction error for China, as that will likely be
    on the order of \$100 billion. Squared log error corrects that, at least somewhat.
    (It is questionable, however, whether we should use the same regressor for Andorra and China -- these
    two countries do not have much in common.)

The list is not exhaustive. Specialised applications may call for specialised loss functions. 
There are, for instance, [error metrics](https://uk.mathworks.com/help/images/image-quality-metrics.html) for image data that try to capture whether 
images *look* differently to humans or not.


Imbalanced classes
------------------

Not all classes will always have the same importance in classification. A prime example is a spam filter: It is merely annoying if a spam email is not 
detected, an important mail, however, that is banished to the spam folder may cause some grief. We'd prefer if the 
spam filter is more cautious when classifying emails as spam. 

A somewhat similar issue may arise when the classes are not present in equal numbers in the training set. Say, we try to classify cat and dog pictures.
Scraping the internet results in 90000 cat pictures but only 10000 dog pictures. For whatever reasons, we may still insist that our classifier perform
equally well for cat and dog pictures -- and that's a problem, as a classifier that always outputs `cat' will likely achieve an accuracy 
of around 90\%. 

To measure the performance of classifiers in a more fine-grained manner a number of metrics have been introduced. Most of these apply
to binary classification. Then, it is customary to designate one class as the *positive* class, and the other as the *negative* class. 
Let's say that *ham* is the positive class, and *spam*, the negative class. 
Given a test set, we then count

* *tp*, the *true positives*, the number of samples of positive class that were correctly classified as positive;

* *fp*, the *false positives*, the number of samples of negative class that were incorrectly classified as positive;

* *tn*, the *true negatives*, the number of samples of negative class that were correctly classified as negative; and

* *fn*, the *false negatives*, the number of samples of positive class that were incorrectly classified as negative.


The simplest metrics are now the detection rates in each class:

* *true positive rate*

    $$
    \frac{tp}{tp+fn},
    $$

    ie, what percentage of ham emails is recognised as ham?
    Depending on the context, the true positive rate may also be called *recall* or *sensitivity*.
    
* *true negative rate* 

    $$
    \frac{tn}{tn+fp},
    $$

    ie, what percentage of spam is recognised as spam?
    The true negative rate is also called *specificity*.

A third type of metric is:

* *precision*

    $$
    \frac{tp}{tp+fp},
    $$

    ie, the rate of actual ham emails among all as ham classified emails. 

It is easy to check that from any two of these metrics we can compute the third (assuming that we know 
how many positive and how many negative samples are present in the dataset).

Going back to the spam filter, it seems that true positive rate should be more important than true negative rate and precision. 
In the example of imbalanced cat and dog pictures, we may aim for true positive and true negative rates of similar
value. 

How now may we influence true positive rate, true negative rate and precision? There are two approaches, one that applies
during training and one that applies after training. Let's do the latter one first.



Trade-off between true positive and false positive rate
-------------------------------------------------------

[^lendingcode]
For binary classification, we can express a different importance of the classes as 
differing preferences for true positive and negative rate. That is, we may specify 
a target true negative rate of 90\% and then optimise the true positive rate as much as possible. 

[^lendingcode]: {-} [{{ codeicon }}lendingclub](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/losses/lendingclub.ipynb)

```{figure} pix/roc.png
:name: rocfig
:height: 6cm

A ROC curve.
```

How can we achieve that? Most classifiers compute more than just the prediction, ie, +1 or -1, or 'cat' or 'dog'.
In almost all cases, the classifier computes a confidence level in $[0,1]$ to express
how likely it is that the current sample belongs to class 1. Even those classifiers
that don't do that, normally output a score in $\mathbb R$, where a positive value 
would indicate class 1, and negative value class -1. In both cases, there is a
*threshold* $t$ such that if the classifier $h$ outputs a value above $t$ for a sample $x$,
ie, if $h(x)>t$, then we'd predict class 1, and class -1 otherwise. 
For a classifier that outputs a confidence level in $[0,1]$ the threshold
would be $t=\tfrac{1}{2}$. 

What happens if we artificially increase $t$? Then it becomes harder for a sample 
to be classified as 1. Consequently, the number of true positives, *tp*, would decrease, 
as would the number of false positives, *fp*. In contrast, the number of true negatives, *tn*,
and the number of false negatives, *fn*,
would increase. As a result, the true positive rate would decrease, while the true negative rate would increase.
 
Thus, we can systematically vary the threshold, compute true positive and negative rates and then 
pick the pair that is most suitable to us.
The plot of the pairs of true positive rate and *false positive rate*, the complement of the 
true negative rate, ie, $1-tnr$, is called *ROC curve*, or *receiver operating characteristic curve*.[^rocname] 

[^rocname]: {-} Why the odd name? According to [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#History), 
ROC curves were invented during World War II in order to 
fine-tune the procedures to detect enemy fighter planes by radar. The receiver probably denotes the radar that captures the 
radar waves.
%https://stats.stackexchange.com/questions/341043/what-is-the-origin-of-the-receiver-operating-characteristic-roc-terminology


A similar approach, systematically increasing the threshold, also allows to pick 
the best balance between precision and recall. The corresponding curve, the precision-recall curve,
however, may be slightly less well behaved. Indeed, while recall never increases with larger threshold, 
precision will occasionally increase even if, typically, it decreases with larger threshold. 

There are two drawbacks to fine-tuning the performance for the different classes 
after the classifier has been trained. First, a minority class may be more or less be overlooked
during training, in which case it is quite hard to force a classifier to pay attention to it afterwards. 
Better, to put some more weight on it already during training. Second, the trade-off between true positive
and negative rate, or between precision and recall is quite straightforward when there are only two 
classes. When there are more, it is less clear how best to strike the balance between the classes. 



Class weights in losses
-----------------------

We return to, potentially, multi-class classification.
Often, the key metric in classification is simply *zero-one loss*, ie 

$$
\ell_{0-1}(y,y')=
\begin{cases}
0 & \text{if }y=y'\\
1 & \text{if }y\neq y'
\end{cases}
$$

When we have classes of differing importance, then misclassifying one class for another may be 
of different seriousness. Sending an important mail to the spam folder is more serious
then letting a Viagra ad pass through the filter. In such a case, we may weigh the loss 
depending on the true class. That is, we specify a weight function $w:\mathcal Y\to\mathbb R_+$,
and define a loss 

$$
\ell^*(y,y')=
\begin{cases}
0 & \text{if }y=y'\\
w(y) & \text{if }y\neq y'
\end{cases}
$$

For the spam filter, we would perhaps set $w(\textsf{ham})=10$ and $w(\textsf{spam})=1$, in order to 
signal that classifying a ham email as spam is ten times as bad as the other way round. 

We note rightaway that

$$
\ell^*(y,y') = w(y) \ell_{0-1}(y,y')\quad\text{for all }y,y'\in\mathcal Y,
$$

which we abbreviate to $\ell^*=w\ell_{0-1}$.


Once we've decided on a loss, we obtain a target quantity, namely the expected loss, 
that should be minimised:

$$
L_{\mathcal D,\ell}(h) = \expec_{(x,y)\sim\mathcal D}[\ell(y,h(x))]
$$

Here, $\mathcal D$ is the distribution on $\mathcal X\times\mathcal Y$ that descibes the 
classification problem, and $h:\mathcal X\to\mathcal Y$ is a classifier.

```{prf:Lemma}
:label: distlem
Let $w:\mathcal Y\to\mathbb R_+$ be a class weighting. Then for every 
distribution $\mathcal D_1$ there is a distribution $\mathcal D_2$ and a constant $Z>0$
such that for all loss function $\ell_1$ and $\ell_2$ with $\ell_1=w\ell_2$
it holds that 

$$
L_{\mathcal D_1,\ell_1}(h) = ZL_{\mathcal D_2,\ell_2}(h)
$$

for every classifier $h:\mathcal X\to\mathcal Y$.
```

````{prf:Proof}
For simplicity, we assume $\mathcal D_1$ to be discrete.
We start calculating
\begin{align*}
L_{\mathcal D_1,\ell_1}(h) & = \sum_{(x,y)\in\mathcal X\times\mathcal Y}\proba_{\mathcal D_1}[(x,y)] \ell_1(y,h(x)) \\
& = \sum_{(x,y)\in\mathcal X\times\mathcal Y}\proba_{\mathcal D_1}[(x,y)]w(y) \ell_2(y,h(x))
\end{align*}
Putting 

$$
Z=\sum_{(x,y)\in\mathcal X\times\mathcal Y}\proba_{\mathcal D_1}[(x,y)]w(y),
$$

we see that $Z>0$ is the normalisation factor that turns  

$$
\proba_{\mathcal D_2}[(x,y)] = \frac{1}{Z}\proba_{\mathcal D_1}[(x,y)]w(y)\quad\text{for }(x,y)\in\mathcal X\times\mathcal Y
$$

into a probability distribution on $\mathcal X\times\mathcal Y$. 

We finish with 
\begin{align*}
L_{\mathcal D_1,\ell_1}(h) & = Z\sum_{(x,y)\in\mathcal X\times\mathcal Y}\proba_{\mathcal D_2}[(x,y)] \ell_2(y,h(x)) \\
& = ZL_{\mathcal D_2,\ell_2}(h)
\end{align*}
Note that $\mathcal D_2$ as well as $Z$ only depend on $\mathcal D_1$ and $w$, but not on $\ell_1$ or $\ell_2$.
````

For later use, we extract from the proof that
```{math}
:label: xD1D2
\proba_{\mathcal D_2}[(x,y)]=\frac{1}{Z}w(y)\proba_{\mathcal D_1}[(x,y)]\quad\text{for all }(x,y)\in\mathcal X\times\mathcal Y,
```
where $Z$ is just a normalisation factor that guarantees that all probabilities add to 1.
We also calculate:
\begin{align*}
\proba_{\mathcal D_2}[y|x]&=\frac{\proba_{\mathcal D_2}[(x,y)]}{\proba_{\mathcal D_2}[x]} 
= \frac{w(y)\proba_{\mathcal D_1}[(x,y)]}{Z}\cdot\frac{1}{\sum_{y'\in\mathcal Y}\proba_{\mathcal D_2}[(x,y')]} \\
& = \frac{w(y)\proba_{\mathcal D_1}[(x,y)]}{Z}\cdot\frac{Z}{\sum_{y'\in\mathcal Y}\proba_{\mathcal D_1}[(x,y')]w(y')} \\
&= \frac{w(y)\proba_{\mathcal D_1}[x]\cdot\proba_{\mathcal D_1}[y|x]}{\sum_{y'\in\mathcal Y}\proba_{\mathcal D_1}[x]\cdot\proba_{\mathcal D_1}[y'|x]w(y')}
= \frac{w(y)\proba_{\mathcal D_1}[y|x]}{\sum_{y'\in\mathcal Y}\proba_{\mathcal D_1}[y'|x]w(y')}
\end{align*}

We get:
```{math}
:label: D1D2
\proba_{\mathcal D_2}[y|x]=\frac{w(y)\proba_{\mathcal D_1}[y|x]}{\sum_{y'\in\mathcal Y}\proba_{\mathcal D_1}[y'|x]w(y')}
```

Why is that interesting? Because it immediately indicates a strategy how to cope with 
a loss function with class weights. Recall the example of the spam filter, where we had 
weights $w(\textsf{ham})=10$ and $w(\textsf{spam})=1$, and a loss function $\ell^*$ that is simply
a weighted version of the zero-one loss. The lemma implies that if we change the distribution 
accordingly then we can minimise zero-one loss, as usual. How do we change the distribution?
We make it 10 times (because $w(\textsf{ham})=10$) more likely that ham emails are picked 
for the training set, which we can simulate by adding 9 copies of every ham email to the training set.


A Bayes-classifier reaches the smallest possible loss. We extend the definition to
arbitrary loss functions: Given a loss $\ell^*$ and a distribution $\mathcal D$
on $\mathcal X\times\mathcal Y$, we call a classifier $h^*$
a *Bayes-classifier for $\mathcal D,\ell^*$* if 

$$
L_{\mathcal D,\ell^*}(h^*)=\inf_h L_{\mathcal D,\ell^*}(h)
$$
 

With the help of the previous lemma, we can compute the Bayes-classifier for the loss $\mathcal D,\ell^*$. 
Because the statement is nicer, we only determine the Bayes-classifier for binary classification.

```{prf:Theorem}
:label: wghbaythm
Let a binary classification problem be defined by a distribution $\mathcal D$ on  $\mathcal X\times \{-1,1\}$,
 let $w(1),w(-1)$ define a class weighting,  and let $\ell^*=w\ell_{0-1}$ be the 
correspoding loss function. Then 

$$
h^*(x) = \begin{cases}
1& \text{ if }\proba_\mathcal D[1|x]\geq \frac{w(-1)}{w(1)+w(-1)}\\
-1 & \text{ otherwise}
\end{cases}
$$

is a Bayes-classifier for $\mathcal D,\ell^*$.
```

As an illustration, consider a picture that has a 10\% chance of showing a cat, and a 90\% chance
of showing a dog. (Why the uncertainty? Think of a very grainy picture, taken at night. From a distance.
By someone with very unsteady hands.) An unweighted Bayes-classifier would output 'dog'. As everybody knows, however, cats 
are ten times as important as dogs. Accordingly, we set weights to $w(\mathsf{dog})=1$ and  $w(\mathsf{cat})=10$,
resulting in a probability threshold of

$$
\frac{w(\mathsf{dog})}{w(\mathsf{cat})+w(\mathsf{dog})}\cdot 100\% = \frac{1}{10+1}\cdot 100\% \approx 9.1\%
$$

Thus, with only a 10\% probability that the grainy picture shows a cat, the weighted Bayes-classifier would still
return 'cat'.

````{prf:Proof}
Let $\mathcal D_2$ and $Z$ be as in {prf:ref}`distlem` for $\mathcal D$, the class weighting $w$
and $\ell^*=w\ell_{0-1}$.
Then

$$
\inf_h L_{\mathcal D,\ell^*}(h) = Z\inf_h L_{\mathcal D_2,\ell_{0-1}}(h)
$$

and, thus, a Bayes-classifier $h^*$ for $\mathcal D_2,\ell_{0-1}$ will be a Bayes-classifier for $\mathcal D,\ell^*$.
Then 

$$
h^*(x)=\begin{cases}
1 & \text{ if }\proba_{\mathcal D_2}[1|x]\geq \tfrac{1}{2}\\
-1 & \text{ otherwise}
\end{cases}
$$

is a Bayes-classifier for $\mathcal D_2,\ell_{0-1}$ and thus also for for $\mathcal D,\ell^*$.

With {eq}`D1D2` we may express this in the original distribution as follows. 
We write $\eta(x)=\proba_{\mathcal D}[1|x]$. Then
\begin{align*}
& \proba_{\mathcal D_2}[1|x]\geq \tfrac{1}{2} \\
\Leftrightarrow\quad & \frac{\eta(x)w(1)}{\eta(x)w(1)+(1-\eta(x))w(-1)} \geq \tfrac{1}{2}\\
\Leftrightarrow\quad & 2\eta(x)w(1) \geq \eta(x)w(1)+(1-\eta(x))w(-1)\\
\Leftrightarrow\quad & \eta(x)\geq \frac{w(-1)}{w(1)+w(-1)}
\end{align*}
````

Bayes consistency and arbitrary classification losses
-----------------------------------------------------

Surrogate loss functions are computationally feasible loss function that can be minimised
instead of zero-one loss. If the loss function is Bayes-consistent, then the minimiser
of the surrogate loss function will be a Bayes-classifier. How do we need to adapt
the surrogate losses if we aim for a different loss than zero-one loss? 

Let $\ell^*:\mathcal Y\times\mathcal Y\to\mathbb R_+$ be a loss function. 
A (surrogate) loss function $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$ 
is *Bayes-consistent for $\ell^*$* if for every distribution $\mathcal D$ on 
$\mathcal X\times\mathcal Y$ and
for every sequence 
$h_1,h_2,\ldots:\mathcal X\to\mathcal Y$ of classifiers with 

$$
\lim_{i\to\infty} L_{\mathcal D,\ell}(h_i) = \inf_h L_{\mathcal D,\ell}(h)
$$
 
it follows that

$$
\lim_{i\to\infty} L_{\mathcal D,\ell^*}(h_i) = \inf_h L_{\mathcal D,\ell^*}(h)
$$
 

```{prf:Theorem}
Let $w:\mathcal Y\to\mathbb R_+$ be a class weighting, and let $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$
be a Bayes-consistent loss function for the zero-one loss $\ell_{0-1}$. 
If $\ell^*=w\ell_{0-1}$ then $\overline\ell=w\ell$ is Bayes-consistent for $\ell^*$.
```

````{prf:Proof}
Let $\mathcal D$ be a distribution on $\mathcal X\times\mathcal Y$, and let $h_1,h_2,\ldots$ be 
a sequence of classifiers $\mathcal X\to\mathcal Y$ with 

$$
\lim_{i\to\infty} L_{\mathcal D,\overline\ell}(h_i) = \inf_h L_{\mathcal D,\overline\ell}(h)
$$

By {prf:ref}`distlem`, there is a distribution $\mathcal D_2$ and a constant $Z>0$ such that 
$L_{\mathcal D,\ell_1}(h)=ZL_{\mathcal D_2,\ell_2}(h)$ for every pair of loss functions
with $\ell_1=w\ell_2$ and every classifier $h:\mathcal X\to\mathcal Y$.

Then, it holds that also 

$$
\lim_{i\to\infty} L_{\mathcal D_2,\ell}(h_i) = \inf_h L_{\mathcal D_2,\ell}(h)
$$

As $\ell$ is Bayes-consistent for $\ell_{0-1}$ it follows that

$$
\lim_{i\to\infty} L_{\mathcal D_2,\ell_{0-1}}(h_i) = \inf_h L_{\mathcal D_2,\ell_{0-1}}(h)
$$

Using the conclusion of {prf:ref}`distlem` again, we obtain that

$$
\lim_{i\to\infty} L_{\mathcal D,\ell^*}(h_i) = \inf_h L_{\mathcal D,\ell^*}(h),
$$

and we see that $\ell$ is Bayes-consistent for $\ell^*$.
````

Let us consider the spam filter again. There, we may fix as target loss

$$
\ell^*(y,y')=\begin{cases}
0 & \text{ if }y=y'\\
1 & \text{ if }y=\textsf{spam},\,y\neq y'\\
10 & \text{ if }y=\textsf{ham},\,y\neq y'
\end{cases}
$$

The logistic loss function

$$
\ell(y,y')=\log\left(1+e^{-1yy'}\right)
$$

is known to be Bayes-consistent for the zero-one loss. The theorem now implies that, for loss $\ell^*$,
we should adapt the logistic loss, and use

$$
\overline\ell(y,y')=w(y)\ell(y,y'),\text{ with }
w(y)=\begin{cases}
1 & \text{ if }y=\textsf{spam}\\
10 & \text{ if }y=\textsf{ham}
\end{cases}
$$

instead.[^imbacode]

[^imbacode]: {-} [{{ codeicon }}imbalanced](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/imbalanced_data.ipynb)

If we train a neural network with [SGD](sgdsec), or one of its variants, then we can calculate the gradients
as for zero-one loss and then, at the end, multiply the gradient with the class weight $w(y)$.

%% class_weights in tensorflow: fit(class_weights=some_weights) -> loss is weigthed accordingly
%% same in scikit-learn, for instance, for RandomForestClassifier.fit
%% TODO: demonstrate use of class_weights in jupyter notebook.
%% https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/imbalanced_data.ipynb






