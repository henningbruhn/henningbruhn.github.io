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

Stochastic gradient descent
===========================

How is a neural network trained? How can we minimise logistic loss in order 
to learn the parameters of a logistic regression? Both cases reduce to an
optimisation problem that requires a numerical optimisation algorithm, often a variant
of a gradient descent technique. In the nicest and simplest setting, a convex optimisation
problem, these are even guaranteed to find an optimal solution.

Convexity
---------

A set $C\subseteq\mathbb R^n$ is *convex* if the connecting segment
between any two points in $C$ is also contained in $C$:

$$
\lambda x+(1-\lambda)y\in C\text{ for all }x,y\in C\text{ and }\lambda\in[0,1].
$$

    
```{figure} pix/convexsets.png
:figwidth: 15cm
:name: convsetfig

A convex set in  (a) and (b); the set in (c) is not convex
```
Let $C\subseteq\mathbb R^n$ be a convex set.
A function $f:C\to\mathbb R$ is a *convex function* if for all $x,y\in C$ and all $\lambda\in[0,1]$
it holds that 

$$
f(\lambda x+(1-\lambda) y)\leq \lambda f(x) + (1-\lambda) f(y)
$$

Obviously, a linear function is always convex. 

```{figure} pix/convexconcave.png
:figwidth: 15cm
:name: convfunfig
    
A convex function, a concave function (negative of a convex function), and a function that is neither convex nor concave.
```

The two notions of convexity, convex sets and convex functions, are related via the 
epigraph of a function. For a function $f:C\to\mathbb R$, the *epigraph*
is defined as

```{math}
:label: convdef
\text{epi}(f)=\{(x,y) : x\in C,y\geq f(x)\}
```

That means, the epigraph is simply the set of all points above the function graph. 

Convex sets and convex functions are now related as follows:

```{prf:Proposition}
Let $C\subseteq\mathbb R^n$ be a convex set, and let $f:C\to\mathbb R$ be a function.
Then $f$ is a convex function if and only if the epigraph $\text{epi}(f)$ is a convex set.
```

Convex optimisation problems
----------------------------

A *convex optimisation problem* is any problem of the form 

```{math}
:label: convopt

\inf f(x),\quad x\in K
```

where $K\subseteq \mathbb R^n$ is a convex set and $f:K\to\mathbb R$
a convex function.

A point $x^*\in K$ is a *local minimum* if there is an open ball $B$
around $x^*$ such that 

$$
f(x^*)\leq f(x) \text{ for all }x\in B\cap K
$$

```{prf:Proposition}
If $x^*$ is a local minimum of {eq}`convopt` then it is also a 
global minimum. 
```
```{prf:Proof}
Suppose there is a $z\in K$ with $f(z)<f(x^*)$. Let $B$ be a ball around
$x^*$ such that $f(x^*)\leq f(x)$ for all $x\in B\cap K$.
Since $K$ is convex,
$x_\lambda=\lambda x^*+(1-\lambda)z\in K$ for all $\lambda\in [0,1]$.
In particular, there is a $\lambda\in (0,1]$ such that $x_\lambda\in B$. 
Because $f$ is convex

$$
f(x_\lambda)\leq \lambda f(x^*)+(1-\lambda)f(z)<f(x^*)
$$

as $\lambda\neq 0$ and $f(z)<f(x^*)$. This, however, is a contradiction to $x^*$
being a local minimum.
```

Note that it makes a difference whether we aim to minimise or maximise a convex 
function over a convex set. Indeed, if we maximise the convex function in {numref}`Figure {number} <convfunfig>`
over the convex set $[0,3]$ we see that $x^*=0$ is a local maximum but not  a 
global one (that would be $x=3$).

Convex functions
----------------

Which functions are convex? 
Norms are convex. Indeed, the function $x\mapsto ||x||$ is convex as for every $\lambda\in [0,1]$
the triangle inequality implies:

$$
||\lambda x+(1-\lambda) y||\leq ||\lambda x|| + ||(1-\lambda) y|| = \lambda||x||+(1-\lambda)||y||
$$

Recall that $\nabla f(x) = \trsp{\left(\frac{\partial f}{\partial x_1}(x),\ldots,\frac{\partial f}{\partial x_n}(x)\right)}$
is the *gradient* of $f$ at $x$.

```{prf:Lemma}
:label: gradlem
Let $f:C\to\mathbb R$ be a differentiable function on an open convex set $C\subseteq \mathbb R^n$. 
Then $f$ is convex if and only if 

$$
f(y)\geq f(x)+\trsp{\nabla f(x)}(y-x)\text{ for all }x,y\in C.
$$

```
```{prf:Proof}
First we do $n=1$, i.e. we prove that 

$$
\text{$f$ is convex} \quad\Leftrightarrow\quad  f(y)\geq f(x)+f'(x)(y-x)\text{ for all }x,y\in C
$$

Assume first that $f$ is convex. Then for every $\lambda\in[0,1]$
\begin{align*}
\lambda f(y) &\geq f(x+\lambda(y-x))-(1-\lambda) f(x)
\end{align*}
We divide by $\lambda$:
\begin{align*}
f(y) &\geq \frac{f(x+\lambda(y-x))-f(x)}{\lambda}+f(x)\\
&=\frac{f(x+\lambda(y-x))-f(x)}{\lambda(y-x)}(y-x)+f(x)\\
&= \frac{f(x+t)-f(x)}{t}(y-x)+f(x)
\end{align*}
for $t=\lambda(y-x)$. Now taking $t\to 0$, we get $f(y)\geq f(x)+f'(x)(y-x)$.

For the other direction, we put $z=\lambda x+(1-\lambda)y$, and obtain 

$$
f(x)\geq f(z)+f'(z)(x-z)\text{ and }f(y)\geq f(z)+f'(z)(y-z)
$$

We multiply the first inequality with $\lambda$, the second with $(1-\lambda)$ and add them.
This finishes the case $n=1$.

For $n>1$, we define $g:[0,1]\to\mathbb R$ by $g(\lambda)=f(\lambda x+(1-\lambda) y)$
and then apply the one-dimensional case. We omit the details.
```


If a function is twice differentiable then whether it is convex can be read off 
its second derivative:

```{prf:Lemma}
:label: twicelem
Let $f:C\to\mathbb R$ be a twice differentiable function
on an open interval $C\subseteq \mathbb R$.  Then the following statements
are equivalent:
1. $f$ is convex;
2. $f'$ is monotonically non-decreasing; and
3. $f''$ is non-negative.
```

Again, I omit the proof. There is also a version for multivariate functions.

As a consequence of the lemma,
$x\mapsto x^2$ is a convex function over $\mathbb R$, and so is $x\mapsto e^x$.
Also, the function $f:x\mapsto \log(1+e^x)$ is convex: Indeed, 

$$
f'(x)=\frac{e^x}{1+e^x}=\frac{1}{1+e^{-x}},
$$

which is monotonically increasing.

Compositions of convex functions are not generally convex: Indeed, both $f:x\mapsto x^2$ and 
$g:x\mapsto e^{-x}$ are both convex, but $g\circ f:x\mapsto e^{-x^2}$ is not. This is different if the 
inner function is affine. 
% see Boyd & Vandenberghe p.83

```{prf:Lemma}
:label: affconflem
Let $g:\mathbb R\to\mathbb R$ be convex, and let $w\in\mathbb R^n$ and $b\in\mathbb R$. 
Then $f(x)=g(\trsp wx+b)$ is also convex.
```
```{prf:Proof}
Let $x,y\in\mathbb R^n$ and $\lambda\in [0,1]$. Then
\begin{align*}
f(\lambda x+(1-\lambda)y) &= g(\lambda (\trsp wx+b) + (1-\lambda)(\trsp wy+b))\\
&\leq \lambda g(\trsp wx+b) + (1-\lambda) g(\trsp wy+b)\\
& = \lambda f(x)+(1-\lambda)f(y),
\end{align*}
as $g$ is convex. 
```

As a consequence, for fixed $x\in\mathbb R^n$, $y\in\mathbb R$ the 
 function $f:\mathbb R^n\to\mathbb R$, $w\mapsto \log(1+e^{-y\trsp wx})$ 
is convex.


The following statement is almost trivial to prove:

```{prf:Lemma}
:label: sumlem
Let $C\subseteq\mathbb R^n$ be a convex set, let $w_1,\ldots, w_m\geq 0$, 
and 
let $f_1,\ldots,f_m:C\to\mathbb R$ be convex functions. Then $f=\sum_{i=1}^mw_if_i$
is a convex function.
```


Recall that  logistic regression works by minimising the logistic loss.
As a consequence of the previous lemmas, we get:

```{prf:Lemma}
:label: loglosslem
For every finite training set $S\subseteq \mathbb R^n\times\{-1,1\}$, 
the logistic loss function

$$
w\mapsto \frac{1}{|S|}\sum_{(x,y)\in S}-\log_2\left(\sigm(y\trsp wx)\right)
$$

is convex.
```

Recall that 

$$
\sigm:z\mapsto \frac{1}{1+e^{-z}}
$$

```{prf:Proof}
We have already seen that the function 
$f:\mathbb R^n\to\mathbb R$, $w\mapsto \log(1+e^{-y\trsp wx})$ 
is convex, for fixed $x\in\mathbb R^n$ and $y\in\{-1,1\}$. 
Now, the logistic loss is simply the sum of such functions, weighted with 
the positive factor $\tfrac{1}{|S|}$:

$$
\frac{1}{|S|}\sum_{(x,y)\in S}-\log_2\left(\sigm(y\trsp wx)\right)
= \frac{1}{|S|}\sum_{(x,y)\in S}\log_2\left(1+e^{-y\trsp wx}\right)
$$

 Thus, it follows from {prf:ref}`sumlem`
that the logistic loss function is convex.
```

Recall that when performing logistic regression we aim to find a linear classifier
with small zero-one loss. Instead of minimising the zero-one loss directly, however, 
we minimise the logistic loss -- which we had seen to upper-bound the zero-one loss; 
see {prf:ref}`upperloglosslem`. Here now is the reason, why we replace the zero-one loss 
by a *surrogate loss* function, the logistic loss: in contrast to zero-one loss, 
the logistic loss function is convex!

Let's look at one more way to obtain a convex function.

```{prf:Lemma}
:label: suplem
Let $I$ be some index set, let $C$ be a convex set. 
Let $f_i:C\to\mathbb R$, $i\in I$, be a family of convex functions. Then 
$f:x\mapsto\sup_{i\in I}f_i(x)$ is a convex function. 
```
```{prf:Proof}
Let $x,y\in C$ and $\lambda\in [0,1]$. Then for every $i^*\in I$,
because $f_{i^*}$ is convex, it holds that:
\begin{align*}
f_{i^*}(\lambda x+(1-\lambda)y) &\leq \lambda f_{i^*} + (1-\lambda) f_{i^*}(y)\\
& \leq \sup_{i\in I}\lambda f_{i} + (1-\lambda) f_{i}(y)
\leq \lambda \sup_{i\in I}f_{i} + (1-\lambda) \sup_{i\in I}f_{i}(y)
\end{align*}
Therefore it also holds that 

$$
\sup_{i\in I}f_{i}(\lambda x+(1-\lambda)y)
\leq \lambda \sup_{i\in I}f_{i} + (1-\lambda) \sup_{i\in I}f_{i}(y)
$$

```

Strong convexity
----------------

Many of the functions we encounter in machine learning are at least locally convex,
and usually these even exhibit a stronger notion of convexity that is called, 
well, *strong* convexity. The difference between convexity and strong convexity
is basically the difference between an affine function such as $x\mapsto x$ and a
quadratic function such as $x\mapsto x^2$. Affine functions are convex but barely so:
they satisfy the defining inequality of convexity {eq}`convdef` with equality. For
a strongly convex function this will never be the case.

A function $f:K\to\mathbb R$ on a convex set $K\subseteq\mathbb R^d$ is 
*$\mu$-strongly convex* for $\mu>0$ if for all $\lambda\in [0,1]$
and $x,y\in K$ it holds that 

$$
\lambda f(x)+ (1-\lambda)f(y)\geq f(\lambda x+(1-\lambda)y) +\frac{\mu}{2}\lambda(1-\lambda)
||x-y||^2_2
$$

Clearly, it is the additional term $\frac{\mu}{2}\lambda(1-\lambda)
||x-y||^2_2$ that makes strong convexity a stronger notion than ordinary convexity. 
In particular, affine functions are convex but not $\mu$-strongly convex for any $\mu>0$. 

```{prf:Lemma}
:label: strongnormlem
The function $\mathbb R^d\to\mathbb R$, $x\mapsto ||x||^2_2$ is 
$2$-strongly convex.
```
```{prf:Proof}
Let $\lambda\in[0,1]$ and $x,y\in\mathbb R^d$. Then

$$
||\lambda x+(1-\lambda)y||^2 = \lambda^2||x||^2+2\lambda(1-\lambda)\trsp xy+(1-\lambda)^2||y||^2
$$

and 

$$
\lambda(1-\lambda)||x-y||^2 = \lambda(1-\lambda)||x||^2-2\lambda(1-\lambda)\trsp xy+\lambda(1-\lambda)||y||^2
$$

Adding the two right-hand sides gives $\lambda ||x||^2+(1-\lambda)||y||^2$.
```

```{prf:Lemma}
:label: stronglem
Let $g:K\to\mathbb R$ be a $\mu$-strongly convex function on a convex set $K\subseteq\mathbb R^d$.
Then
1. $Cg$ is $C\mu$-strongly convex for any $C>0$; and
2. if $f:K\to\mathbb R$ is convex then $f+g$ is $\mu$-strongly convex.
```
```{prf:Proof}
1. is trivial and so is 2.
```

Here, statement 2. is the reason why strong convexity is relevant to us.
Often, we might have a convex loss function $L(w)$ and then add a term $\mu||w||_2$ 
to the loss function that penalises large weights. This is a common strategy, called *regularisation*,
that we will treat later. A fortunate consequence is then that the new function $w\mapsto L(w)+\mu||w||_2^2$
is even strongly convex.

```{prf:Lemma}
:label: strongdifflem
Let $f:K\to\mathbb R$ be a differentiable
 function on an open convex set $K\subseteq\mathbb R^d$.
Then $f$ is $\mu$-strongly convex if and only if for all $x,y\in K$

$$
f(y)\geq f(x)+\nabla \trsp{f(x)} (y-x)+\frac{\mu}{2}||y-x||^2
$$
```
The proof is an obvious modification of the proof of {prf:ref}`gradlem`.

We draw a simple consequence. If $x$ is a global minimum of $f$ then,
as $\nabla f(x)=0$ it follows that 

```{math}
:label: strongmin2
f(y)-f(x)\geq \frac{\mu}{2}||y-x||^2
```

Gradient descent
----------------

% IN Boyd \& Vandenberghe: convergence rates for strongly convex functions (those with 
% constant min curvature), non-constant learning rate, namely line search or backtracking.
% no discussion of SGD
% 
% In Deep learning: short discussion of SGD and in particular computational costs 
% on page 148. Also: "gradient descent ... has often been regarded as slow and unreliable", 
% but in fact very useful for ML.


Some of the objective functions in machine learning are convex. 
How can we minimise them? With *stochastic gradient descent* -- it is this 
algorithm (or one of its variants) that powers most of machine learning. Let's understand
simple *gradient descent* first.

```{prf:Algorithm} gradient descent
:label: gdalg
**Instance** A differentiable function $f:\mathbb R^n\to\mathbb R$, a first point $x^{(1)}$.\
**Output** A point $x$.

1. Set $t=1$. 
2. **while** stopping criterion not satisfied:
3.   {{tab}}Compute $\nabla f(x^{(t)})$.
4.   {{tab}}Compute learning rate $\eta_t$.
5.   {{tab}}Set $x^{(t+1)}=x^{(t)}-\eta_t\nabla f(x^{(t)})$.
6.   {{tab}}Set $t=t+1$.   
7. **output** $x^{(t)}$, or best of $x^{(1)},\ldots, x^{(t)}$, or average.
```

There[^gdcode] are different strategies for the learning rate $\eta_t$ (which should always be positive). 
The easiest is a constant 
learning rate $\eta_t=\eta>0$ for all $t$. The problem here is that at the beginning 
of gradient descent, a constant learning rate will probably lead to slow progress,
while near the minimum, it might lead to overshooting. More common are decreasing or
adaptive learning rates, see below. 

[^gdcode]: {-} [{{ codeicon }}gradient descent](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/stochastic_gradient_descent/gradient.ipynb)

Typical stopping criteria are: a pre-fixed maximum number of iterations has been reached;
or the norm of the gradient has become very small. 

Concerning the output: rather than outputting the last $x^{(t)}$ it seems that it cannot hurt 
to output the best $x^{(t)}$ encountered during the execution -- that, however, necessitates
a function evaluation $f(x^{(t)})$ in every step, which can be computationally costly. From a theoretical point
of view, the average $\tfrac{1}{T}\sum_{t=1}^Tx^{(t)}$ is sometimes convenient. 

```{figure} pix/gd_etas.png
:name: etafig
:width: 15cm

Gradient descent with constant learning rates of different values.
The function to be minimised is
$(x,y)\mapsto \tfrac{1}{2}(x^2+10y^2)$. Middle: small learning
rate leads to slow convergence. Right: learning rate is too large, no
convergence.
```

```{prf:Theorem}
If $f:\mathbb R^n\to \mathbb R$ is a convex and differentiable function and certain additional but
mild conditions are satisfied, in particular with respect to the learning rate, then
{prf:ref}`gdalg` will converge towards the global minimum:

$$
x^{(t)}\to x^*,\text{ as }t\to\infty,
$$
 
where $x^*$ is a global minimum of $f$.
```

The statement is intentionally vague. Indeed, there are a number of such results, each with its own set of 
specific conditions. The main point is: For a convex function gradient descent will normally converge. 
We will not discuss this in more detail as plain gradient descent is almost never used in machine learning.

````{dropdown} Gradient descent – an old technique
:color: success
:icon: telescope

```{image} pix/cauchy_methode_generale.png
:width: 10cm
:align: left
```

Gradient descent was invented long before the first neural network
was trained. In the 19th century, the emminent French mathematician
Augustin Louis Cauchy (1789–1857) was studying orbital motions that
are described by an equation in six variables, three variables for
the position of the celestial body in space, and three for its
momentum. As a general method to minimise such equations, Cauchy
proposed a procedure that eventually became known as gradient
descent.

Cauchy contributed to many areas of mathematics and counts as one of
the most prolific mathematicians of all time.

*Méthode générale pour la résolution des systèmes d’équations
simultanées*, A.L. Cauchy (1847)
````

(sgdsec)=
Stochastic gradient descent
---------------------------

Gradient descent is a quite efficient algorithm. Under mild assumptions and with the right 
(adaptable) learning rate it can be shown that the error $\epsilon$, the difference $f(\overline x)-f(x^*)$,
decreases exponentially with the number of iterations, i.e. that

$$
\log(1/\epsilon) \sim t \leftrightarrow \epsilon \sim e^{-t}
$$

Why is gradient descent not normally used in machine learning? Let's consider logistic regression, 
where we have the logistic loss function 

$$
w\mapsto \frac{1}{|S|}\sum_{(x,y)\in S}\log_2\left(1+e^{-y\trsp wx}\right)
$$

Note that we see that loss function as a function on the weights $w$.
To apply gradient descent we first compute the gradient of the loss function $L$ as

$$
\nabla L(w) = \frac{1}{\ln 2\cdot |S|}\sum_{(x,y)\in S}\frac{-y}{1+e^{y\trsp wx}} x
$$

We see that the computation of $\nabla L(w)$ needs $\bigOmega(|S|)$ many operations,
and that thus one iteration of gradient descent has running time $\bigOmega(|S|)$.
In machine learning, data sets may be very large, so large that it becomes infeasible 
to execute many iterations of gradient descent. In such cases, *stochastic gradient descent* (SGD)
starts to shine: instead of computing the full gradient, the gradient with respect to a 
random sample in the training set is computed:

$$
\nabla L_{(x,y)}(w) = \frac{1}{\ln 2} \frac{-y}{1+e^{y\trsp wx}} x
$$
 
Clearly, this is much faster -- it only needs $\bigO(1)$ many operations (we treat the dimension $n$ of
the sample space as a constant). Now, the sample based gradient $\nabla L_{(x,y)}(w)$
will in general by different from the full gradient $\nabla L(w)$. However, its *expectation*
$\expec_{(x,y)}[\nabla L_{(x,y)}(w)]$ coincides with $\nabla L(w)$. This is
the key point exploited by SGD. Moreover, this can be generalised to arbitrary loss
function, or empirical risks.[^sgdcode]

[^sgdcode]: {-} [{{ codeicon }}sgd](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/stochastic_gradient_descent/sgd.ipynb)


```{prf:Algorithm} stochastic gradient descent
:label: sgdalg
**Instance** An empirical risk $L(w)=\tfrac{1}{|S|}\sum_{(x,y)\in S}L_{(x,y)}(w)$, a  point $w^{(1)}$.\
**Output** A point $w$.

1. Set $t=1$.
2. Initialise $w^{(1)}$ to some value.
3. **while** stopping criterion not satisfied:
4. {{tab}}Sample $z_t=(x_t,y_t)$ uniformly from $S$.
4. {{tab}}Compute $\nabla L_{z_t}(w^{(t)})$.
4. {{tab}}Compute learning rate $\eta_t$.
4. {{tab}}Set $w^{(t+1)}=w^{(t)}-\eta_t\nabla L_{z_t}(w^{(t)})$.
4. {{tab}}Set $t=t+1$.
4. **output** $w^{(t)}$, or best of $w^{(1)},\ldots, w^{(t)}$, or average.
```

```{figure} pix/sgd_three_runs.png
:name: setafig
:width: 15cm

Three runs of SGD with same constant learning rate. The function to be minimised 
is $(x,y)\mapsto \tfrac{1}{2}(x^2+10y^2)$, which is not of the type that is typical in machine learning. 
To simulate SGD a normally distributed error is added to each gradient.
```

In a similar way as gradient descent, 
SGD will also converge towards the global minimum 
of a convex loss function. We'll prove a corresponding result below.


:::{dropdown} Learning rates
:color: success
:icon: telescope

How well SGD resolves an optimisation problem depends quite heavily on the learning rates. 
A too small learning rate leads to slow convergence, while a too large rate might even 
preclude convergence. Often, therefore, learning rate schemes are  used that initially have
a large learning rate that, over time, is decreased. In this way, the algorithm takes large 
steps at the beginning and then ever smaller step to home in on some (local) optimum. 

Different learning rate schemes are popular, among them:

* *exponential scheduling:* in iteration $t$, the learning rate is set to
$
\eta_t=\eta_0\cdot 10^{-\frac{t}{r}},
$
where $\eta_0$ and $r>0$ are parameters that need to be fine-tuned.
* *power scheduling:* the learning rate is set to 
$
\eta_t=\eta_0(1+\tfrac{t}{r})^{-c},
$
where again $\eta_0,c,r$ are parameters.
* *performance scheduling:* the learning rate $\eta_t$ is adapted 
with respect to how some performance measure improves. That is, if accuracy (or some other relevant 
measure) stops improving then the learning rate is decreased.


*An empirical study of learning rates in deep neural networks for speech recognition*, 
A. Senior, G. Heigold, M. Ranzato and K. Yang (2013) [link](https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40808.pdf)
:::

Analysis of SGD
---------------

If the loss function is convex then SGD converges, at least in expectation,
towards the global minimum, provided some additional mild conditions are satisfied.[^Tur]
There are a number of such convergence proofs, each with their own set of additional 
conditions. We assume here strong convexity.

[^Tur]: Based on *The convergence of the Stochastic Gradient Descent (SGD) : a self-contained proof*
by G. Turinci, [arXiv:2103.14350](https://arxiv.org/pdf/2103.14350.pdf)


Let $S$ be a training set, and let 

$$
L:w \mapsto\frac{1}{|S|}\sum_{(x,y)\in S}L_{(x,y)}(w)
$$

be a differentiable loss function such that 

1. [$w\mapsto L(w)$ is $\mu$-strongly convex;]{#coni} and 
2. [there is a constant $B$ such that $\sup_{w}||\nabla L_{(x,y)}(w)||^2\leq B$ for all $(x,y)\in S$.]{#conii}

How serious are these assumptions? If the loss function is convex then it is often 
also strongly convex --- due to regularisation. (We'll discuss this later.)
The loss function of a neural network, however, will almost never be convex. 
The second assumption is also a serious restriction. In practice, however, we normally do not
encounter very large gradients, or if we do, we would cap the gradient or
restart the algorithm. 

Let $w^*$ be the global minimum of $L(w)$:

$$
w^*=\argmin_w L(w)
$$


We denote by  $\expec_{1..t}[Z]$ the expected value of some random variable over 
the iterations $1,\ldots, t$ of the SGD algorithm.

Set $\epsilon_t=\expec_{1..t-1}\left[||w^{(t)}-w^*||^2\right]$. Our aim is to show 
that $\epsilon_t\to 0$ as $t\to\infty$ provided the learning rate is adapted in the right way.

We now fix what we need from the learning rate. We'll require 
 positive $\eta_1\geq \eta_2\geq\ldots$
such that 

$$
\sum_{t=1}^\infty\eta_t=\infty\text{ but }\sum_{t=1}^\infty\eta_t^2<\infty
$$

An example here would be a learning rate of the form

$$
\eta_t=\tfrac{\eta_0}{t}
$$



We now prove that SGD is expected to converge towards the minimum. 

```{prf:Theorem}
:label: sgdthm
Let $S$ be a training set, and let 

$$
L:w \mapsto\frac{1}{|S|}\sum_{(x,y)\in S}L_{(x,y)}(w)
$$

 be a differentiable $\mu$-strongly convex loss function such that 
 there is a constant $B$ with
$\sup_{w}||\nabla L_{(x,y)}(w)||^2\leq B$ for all $(x,y)\in S$.
If $\eta_1\geq \eta_2\geq\ldots$ are
such that 
$\sum_{t=1}^\infty\eta_t=\infty$  but $\sum_{t=1}^\infty\eta_t^2<\infty$
then

$$
\expec_{1..t}\left[||w^{(t+1)}-w^*||^2\right]\to 0\text{ as }t\to\infty,
$$

where $w^*$ is a global minimum of $L$.
```

We need a key lemma for the proof:

```{prf:Lemma}
:label: sgdlem4

$$
\epsilon_{T'+k+1}\leq \epsilon_{T'}\prod_{t=T'}^{T'+k}(1-\eta_t\mu)+\sum_{t=T'}^{T'+k}\eta_t^2B
$$

as long as $1-\eta_t\mu\geq 0$ for all $t\in\{T',\ldots, T'+k\}$.
```

With the help of the lemma, we can do the proof of {prf:ref}`sgdthm`:
````{prf:Proof} 
Fix some arbitrary $\delta>0$. We show that then  for every 
large enough  $T$ it holds  that $\epsilon_T\leq\delta$.

First, as $\sum_{t=1}^\infty\eta_t^2<\infty$ there is a $T'$ such that 
```{math}
:label: blubb1
B\sum_{t=T'}^\infty \eta_t^2\leq\tfrac{1}{2}\delta
```
At the same time we can choose $T'$ large enough such that $1-\eta_t\mu\geq 0$ for all $t\geq T'$. 

Next, choose $k_0$ large enough such that 
```{math}
:label: blubb2
\text{exp}\left(\sum_{t=T'}^{T'+k_0}\eta_t\mu\right)\geq \frac{2\epsilon_{T'}}{\delta}
```
This is possible as $\sum_{t=1}^\infty\eta_t=\infty$.
Also note that then {eq}`blubb2` holds for every integer $k\geq k_0$.

With {prf:ref}`sgdlem4` we get for every $T=T'+k+1$ with $k\geq k_0$
\begin{align*}
\epsilon_T &\leq \epsilon_{T'}\prod_{t=T'}^{T'+k}(1-\eta_t\mu)+\sum_{t=T'}^{T'+k}\eta_t^2B\\
&\leq \epsilon_{T'}\prod_{t=T'}^{T'+k}e^{\ln(1-\eta_t\mu)} + \sum_{t=T'}^{\infty}\eta_t^2B\\
& \leq \epsilon_{T'}\text{exp}\left(\sum_{t=T'}^{T'+k}\ln(1-\eta_t\mu)\right) +\tfrac{1}{2}\delta,
\end{align*}
by {eq}`blubb1`. Next, we use $\ln(1-x)\leq -x$, which holds for any $x<1$:
\begin{align*}
\epsilon_T 
& \leq \epsilon_{T'}\text{exp}\left(\sum_{t=T'}^{T'+k}-\eta_t\mu\right) +\tfrac{1}{2}\delta,
\end{align*}
and then {eq}`blubb2` to obtain
$
\epsilon_T \leq \tfrac{1}{2}\delta+\tfrac{1}{2}\delta=\delta.
$
````

It remains to prove {prf:ref}`sgdlem4`. 
As a first step, we prove:
```{prf:Lemma}
:label: sgdlem3

$$
\epsilon_{t+1}\leq\epsilon_t(1-\eta_t\mu)+\eta_t^2B
$$
```
````{prf:Proof}
We start with
\begin{align*}
\epsilon_{t+1} &= 
\expec_{1..t}\left[||w^{(t+1)}-w^*||^2\right] \\ 
& = \expec_{1..t}\left[||w^{(t)}-\eta_t\nabla L_{z_t}(w^{(t)})-w^*||^2\right] \\
& = \epsilon_t -2\eta_t\expec_{1..t}\left[\trsp{(w^{(t)}-w^*)}\nabla^{(t)}\right]+\eta_t^2 \expec_{1..t}[||\nabla^{(t)}||^2], 
\end{align*}
where we have abbreviated $\nabla L_{z_t}(w^{(t)})$ to $\nabla^{(t)}$.
With [condition 2](conii) on $L$, we obtain
```{math}
:label: sgd1
\epsilon_{t+1} \leq \epsilon_t -2\eta_t\expec_{1..t}\left[\trsp{(w^{(t)}-w^*)}\nabla^{(t)}\right]+\eta_t^2 B
```
We focus on the summand in the middle:
\begin{align*}
\expec_{1..t}\left[\trsp{(w^{(t)}-w^*)}\nabla^{(t)}\right] & = 
\expec_{1..t-1}\left[\expec_{z_t}[\trsp{(w^{(t)}-w^*)}\nabla^{(t)}]\right] \\
& = \expec_{1..t-1}\left[\trsp{(w^{(t)}-w^*)}\expec_{z_t}[\nabla^{(t)}]\right] \\
& = \expec_{1..t-1}\left[\trsp{(w^{(t)}-w^*)}\nabla L(w^{(t)})\right]
\end{align*}
We use {prf:ref}`strongdifflem`, to see that

$$
\trsp{(w^{(t)}-w^*)}\nabla L(w^{(t)}) \geq L(w^{(t)})-L(w^*) + \tfrac{\mu}{2}||w^{(t)}-w^*||^2,
$$

which we plug right in:
\begin{align*}
\expec_{1..t}\left[\trsp{(w^{(t)}-w^*)}\nabla^{(t)}\right] & \geq 
\expec_{1..t-1}\left[
L(w^{(t)})-L(w^*) + \tfrac{\mu}{2}||w^{(t)}-w^*||^2 
\right] \\
& \geq 
\expec_{1..t-1}\left[
\tfrac{\mu}{2}||w^{(t)}-w^*||^2 
\right] =  \tfrac{\mu}{2}\epsilon_t,
\end{align*}
as $L(w^*)\leq L(w^{(t)})$ by choice of $w^*$.

We use that in {eq}`sgd1`:

$$
\epsilon_{t+1} \leq \epsilon_t -2\eta_t\cdot\tfrac{\mu}{2}\epsilon_t
+\eta_t^2 B,
$$

which finishes the proof of the lemma.
````

Let's finish the proof of {prf:ref}`sgdlem4`.
```{prf:Proof}
For $k=0$, this is just {prf:ref}`sgdlem3`. Now, we do induction and use {prf:ref}`sgdlem3`
again:
\begin{align*}
\epsilon_{T'+k+1}&\leq \epsilon_{T'+k}(1-\eta_{T'+k}\mu)+\eta_{T'+k}^2B\\
&\leq (1-\eta_{T'+k}\mu)\left(\epsilon_{T'}\prod_{t=T'}^{T'+k-1}(1-\eta_t\mu)+\sum_{t=T'}^{T'+k-1}\eta_t^2B\right)
+\eta_{T'+k}^2B \\
& \leq \epsilon_{T'}\prod_{t=T'}^{T'+k}(1-\eta_t\mu)+\sum_{t=T'}^{T'+k}\eta_t^2B
\end{align*}
as $(1-\eta_{T'+k}\mu)\leq 1$ and $(1-\eta_{T'+k}\mu)\geq 0$.
```

Discussion of SGD
-----------------

Descent methods are old, simple and have apparently been observed
to be ``slow and unreliable''.[^slow] Moreover, 
in convex optimisation, when both algorithms are known to converge,
SGD has even worse convergence rates than vanilla gradient descent. 
In fact, while the error $\epsilon$, the difference $f(x^{(t)})-f(x^*)$ 
is known to drop exponentially for gradient descent, ie 

$$
\log(1/\epsilon)\sim t \text{ or }\epsilon\sim e^{-t}
$$

the error decreases much more slowly for SGD, namely

$$
\epsilon \sim \tfrac{1}{\sqrt t} 
$$

See Bottou[^Bot12]
for more details.

[^slow]: *Deep learning*, p. 148
[^Bot12]: *Stochastic Gradient Descent Tricks*, L. Bottou (2012) and *Online Learning and Stochastic Approximations*, L. Bottou (1998)

% discussion taken/inspired from 8.1.3 of Deep Learning

Still, SGD powers much of machine learning. Well not quite. What 
is used in fact is *minibatch stochastic gradient descent*. 
Somewhat confusingly, gradient descent as discussed above is often referred
to as *batch* gradient descent -- probably because the whole *batch*
of samples is used to compute the gradient. SGD, in contrast, is 
sometimes called *online* stochastic gradient descent, because the 
samples are fed one after another into the algorithm. Finally, in *minibatch*
SGD in each iteration a small sample, say of size 32, or 128, of 
the training set is randomly chosen and then the gradient is computed with 
respect to this sample. 

To be more precise, as in [Section Analysis of SGD](sgdsec) we assume 
the loss function to be 
 $L(w)=\tfrac{1}{|S|}\sum_{(x,y)\in S}L_{(x,y)}(w)$, where $S$ is the training set. 
Then, in each iteration a minibatch $M\subseteq S$ is chosen and the gradient

$$
\nabla L_M(w^{t)}) = \frac{1}{|M|} \sum_{(x,y)\in M} \nabla L_{(x,y)}(w^{(t)})
$$

is computed. Note that this minibatch gradient has the exact same form as the full 
(batch) gradient $\nabla L$. 

The advantage of a minibatch is that it provides a more accurate estimation of the 
full gradient than in (online) SGD. 
Moreover, modern hardware can quite efficiently handle the computation 
of a minibatch gradient in parallel, so that drawing only one sample would 
waste machine efficiency. 

But why now are modern neural networks trained with minibatch SGD, and not
with (batch) gradient descent? Recall that, while we minimise the training error, 
our real aim is to achieve a small generalisation error $L_\mathcal D(w)$. 
In this way, we may see batch gradient, minibatch gradient and online gradient 
all as approximations of the *true* gradient, the gradient that points
in the (opposite) direction of smaller generalisation error:

$$
\expec_{(x,y)\sim\mathcal D}\left[\nabla L_{(x,y)}(w)\right]
$$
 
Recall that, if $X_1,\ldots, X_n$ are iid random variables with variance $\sigma^2$
then their mean has variance $\sigma^2/n$, and also recall *Chebyshev's inequality*:

$$
\proba\left[|X-\mu|\geq\tfrac{\sigma}{\sqrt{\epsilon}}\right]\leq \epsilon,
$$

where $X$ is a random variable, $\mu$ its expectation, $\sigma^2$ the variance and $\epsilon>0$.
(In this form the inequality can be read as: with high probability $X$ deviates from its expectation
by at most $\sigma/\sqrt\epsilon$.)

Now let's apply that to the minibatch gradient. Denote with 
 $\sigma^2$ the variance of (some entry of the vector) $\nabla L_{(x,y)}$. 
Then a minibatch gradient of size $n$ has variance $\sigma^2/n$, which means that Chebyshev's inequality
yields 

$$
\proba\left[|\nabla L_M(w)-\mu|\geq\tfrac{\sigma}{\sqrt{n\epsilon}}\right]\leq \epsilon,
$$

We see that increasing the minibatch size (perhaps even up to the full size of the training set),
yields sublinear returns: the difference of $\nabla L_M(w)$ to the true gradient shrinks only with $1/\sqrt{n}$. 
The time to compute a minibatch gradient based on $n$ samples, however, grows with $n$.

That means minibatch SGD can perform several iterations with a nearly as accurate gradient
in the time batch gradient descent performs a single iteration. Consequently, minibatch SGD
can move more quickly towards a smaller loss. 

There are also other advantages of (minibatch) SGD over gradient descent. For instance, 
the noisier gradient may make it less likely that the algorithm gets trapped in a local minimum.[^lrate]

[^lrate]: {-} [{{ codeicon }}lrate](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/stochastic_gradient_descent/lrate.ipynb)

```{figure} pix/sgd_time.png
:name: gdtimefig
:width: 15cm

Comparison of  logistic loss per running time for batch gradient descent, 
and online and minibatch gradient descent. 
All algorithms were applied to an artificial logistic regression problem. Minibatch size was equal 
to 20, total sample size was 5000. Online and minibatch SGD converge much faster than 
batch gradient descent. Note the different scales of the axes.
```



If the training set is very large then picking a random minibatch in every iteration would actually 
be prohibitively expensive: in order to pick random samples it must be possible to at least access
the whole dataset, which is computationally costly if the dataset is large. A common strategy around this 
is to randomly shuffle the dataset once before SGD is started and then to take always 
consecutive parts of the training set as minibatches. One pass through the whole dataset 
is called an *epoch* -- the number of epochs is often a parameter when training a neural network.
Once the epoch is over, the minibatches repeat. That is, the same 32 (or 128 or whatever) samples
are chosen as a minibatch to compute the gradient. While, from a stochastic point of view, it
would be more proper to shuffle the dataset again after an epoch, this is often avoided because
of the computational burden.[^gdtimes]

[^gdtimes]: {-} [{{ codeicon }}gdtimes](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/stochastic_gradient_descent/gdtimes.ipynb)



In practice the simple version of SGD that I have presented is modified
in various ways. In particular, the gradients and the learning rate are often modified. 
Common, and more efficient, variants of SGD are *Nesterov accelerated gradient*,
*AdaGrad*, *RMSProp* and *Adam*.



