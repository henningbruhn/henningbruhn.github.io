$\newcommand{\bigO}{O}
\newcommand{\trsp}[1]{#1^\intercal} % transpose
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\DeclareMathOperator*{\vari}{\mathbb{V}}   % Probability
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\sigm}{\phi_{\text{sig}}} % logistic function
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
:figwidth: 12 cm
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
:figwidth: 12 cm
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
see Lemma~\ref{upperloglosslem}. Here now is the reason, why we replace the zero-one loss 
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

```{prf:Algorithm}
:label: gdalg
**Instance** A differentiable function $f:\mathbb R^n\to\mathbb R$, a first point $x^{(1)}$.\
**Output** A point :math:$x$.

1. Set $t=1$. 
2. While stopping criterion not satisfied:

   3. Compute $\nabla f(x^{(t)})$.
   4. Compute learning rate $\eta_t$.
   5. Set $x^{(t+1)}=x^{(t)}-\eta_t\nabla f(x^{(t)})$.
   6. Set $t=t+1$.
   
7. Output $x^{(t)}$, or best of $x^{(1)},\ldots, x^{(t)}$, or average.
```

There[^gdcode] are different strategies for the learning rate $\eta_t$ (which should always be positive). 
The easiest is a constant 
learning rate $\eta_t=\eta>0$ for all $t$. The problem here is that at the beginning 
of gradient descent, a constant learning rate will probably lead to slow progress,
while near the minimum, it might lead to overshooting. More common are decreasing or
adaptive learning rates, see below. 

[^gdcode]: {-} [gradient descent](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/stochastic_gradient_descent/gradient.ipynb)

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
:icon: rocket

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

