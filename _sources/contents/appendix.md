$\newcommand{\trsp}[1]{#1^\intercal} % transpose
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\DeclareMathOperator*{\vari}{\mathbb{V}}   % Probability
$


Appendix
========


Very basic probability theory
-----------------------------


A finite probability space  consists of 
of a finite *sample space* $\Omega$ and a *probability measure* $\proba:2^\Omega\to [0,1]$
that satisfies the following properties:

1. $\proba[\Omega]=1$; and

2. $\proba[A]=\sum_{\omega\in A}\proba[\{\omega\}]$ for all $A\subseteq\Omega$.


Let $A,B\subseteq\Omega$ be two sets of elementary events. Then the *conditional probability of $A$ given $B$* 
measures the probability that $A$ occurs when we already 
know that $B$ has occurred. Formally, it is defined as

$$
\proba[A|B]=\frac{\proba[A\cap B]}{\proba[B]},
$$

as long as $\proba[B]>0$. If $\proba[B]=0$ then also $\proba[A|B]=0$.

A *random variable* is simply any function $X\to\mathbb R$. We define 
the *expected value of $X$* as 

$$
\expec[X]=\sum_{\omega\in\Omega}\proba[\{w\}]\cdot X(\omega).
$$

The key property to note is the linearity of expectation, that is

$$
\expec[\lambda X+\mu Y]=\lambda\expec[X]+\mu\expec[Y],
$$

for two random variables $X,Y$ and reals $\lambda,\mu\in\mathbb R$.

An important concept is (stochastic) *independence*: two events $A,B\subseteq\Omega$
are independent if $\proba[A\cap B]=\proba[A]\cdot\proba[B]$. We extend the definition to 
random variables $X,Y$ in saying that $X$ and $Y$ are *independent random variables*
if 

$$
\proba[X\leq a \text{ and }Y\leq b]=\proba[X\leq a]\cdot \proba[Y\leq b]\text{ for all }a,b\in\mathbb R
$$

Basic calculation allows to express the expectation of the product of two 
independent random variables as 

$$
\expec[XY]=\expec[X]\cdot\expec[Y]
$$



Let $A_1,\ldots, A_n\subseteq\Omega$ be events. Then we can upper-bound the probability
of $\bigcup_{i=1}^nA_i$ by 

$$
\proba\left[\bigcup_{i=1}^nA_i\right]\leq\sum_{i=1}^n\proba[A_i]
$$

This bound is obvious but also incredibly useful, so much that it has a proper name: the *union bound*.


%Assume we have a probability distribution $\mathcal D$ over $\mathcal X\times\mathcal Y$. Then 
%the \defi{marginal distribution} over $\mathcal X$ is 
%
$$
%\proba[]
%$$





A simple but often used tool is Markov's inequality:

```{prf:Theorem} Markov's inequality
:label: markovthm

Let $X:\Omega\to\mathbb R_+$ be a random variable with only non-negative values,
and let $t>0$ be a real. Then

$$
\proba[X\geq t]\leq\frac{\expec[X]}{t}
$$
```

We also note a form of *Jensen's inequality*:
% see wikipedia

```{prf:Theorem} Jensen's inequality
:label: jensenexpecthm

Let $X$ be a real-valued  random variable, and let $f:\mathbb R\to\mathbb R$ be a convex function. 
Then 

$$
f(\expec[X])\leq \expec[f(X)]
$$

```

We can not only condition the probability on some event but also the expectation. 
The *conditional expectation* is defined as:

$$
\expec[X|Y=y]=\sum_{\omega}X(\omega)\,\proba[\{\omega\}|Y=y]
$$


A useful fact about conditional expectation is the *law of total expectation*: 
the conditioning vanishes if we wrap it in another expectation:

```{prf:Theorem} Law of total expectation
:label: totalexp

$$
\expec[\expec[X|Y=y]]=\expec[X]
$$

```

The (univariate)[^univariate] *normal distribution* $\mathcal N(\mu,\sigma^2)$ with mean $\mu\in\mathbb R$ and variance $\sigma^2\in\mathbb R$
is defined by the probability density function

$$
p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$
 
That is, the probability that $z\in[a,b]$ is equal to 

$$
\proba[z\in[a,b]] = \int_{[a,b]} \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx
$$

[^univariate]: {-} *univariate* just means: *one variable*, ie, a 1-dimensional function. If we have a distribution in higher
dimension then it's *multivariate*.

The normal distribution with mean $\mu=0$ and variance $\sigma^2=1$ is called the *standard normal distribution* $\mathcal N(0,1)$. 

```{figure} pix/stdnormalfill.png
:name: stdnormalfig
:width: 12cm

The univariate standard normal distribution. Grey area: The probability that the outcome is in $[1.2,1.5]$.
```



Vectors can also be normally distributed; we then talk about a *multivariate normal distribution*. 
We use here only multivariate normal distributions $\mathcal N(\mu,s)$, with $\mu,s\in\mathbb R^n$, that come about as a product
of univariate normal distributions. These have probability density function

$$
\frac{1}{(2\pi)^{n/2}\prod_{i=1}^n\sqrt s_i} e^{-\sum_{i=1}^n\frac{(x_i-\mu_i)^2}{2s_i}}
$$


