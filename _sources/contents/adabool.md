---
orphan: true
---

$\DeclareMathOperator{\sgn}{sgn}$

(adafailsec)=
AdaBoost cannot fit every Boolean function
------------------------------------------

AdaBoost is a type of additive model that combines weak classifiers in a weighted sum.
We denote by $\mathcal W$ the set of weak classifiers. Often the set $\mathcal W$ consists
of *decision stumps*, ie, of classifiers that are defined by a dimension $i\in\{1,\ldots, n\}$, 
a threshold $t\in\mathbb R$ and a sign $\sigma\in\{-1,1\}$: 

$$
x\mapsto \begin{cases}
\sigma & \text{ if }x_i\leq t\\
-\sigma &\text{ otherwise} 
\end{cases}
$$

Let us define as $\mathcal A$ the set of additive models

$$
\mathcal A=\{\sgn\left(\sum_{i=1}^k\alpha_ih_i: \alpha_i\in\mathbb R,\, h_i\in\mathcal W\right) \}
$$

Thus, every classifier learnt by AdaBoost is in the family $\mathcal A$. 

```{prf:Lemma}
:label: fiftylem
:nonumber:

Let $S$ be a training set, and let $h_1,\ldots,h_k\in\mathcal W$ s.t.

$$
L_S(h_i)=0.5\text{ for }i=1,\ldots, k
$$

Then for all $\alpha_1,\ldots,\alpha_k\in\mathbb R$

$$
L_S\left(\sgn\left(\sum_{i=1}^k\alpha_ih_i\right)\right)>0
$$
```

````{prf:Proof}
Let $\alpha_1,\ldots,\alpha_k\in\mathbb R$, and let 

$$
A:=\sgn\left(\sum_{i=1}^k\alpha_ih_i\right)
$$

be the classifier in question.

For $i\in\{1,\ldots, k\}$ it holds that
\begin{align*}
& \tfrac{1}{2}  = L_S(h_i) = \frac{1}{|S|}\sum_{(x,y)\in S}\tfrac{1}{2}(1-yh_i(x)) \\ 
\Leftrightarrow & \sum_{(x,y)\in S}yh_i(x)=0
\end{align*}
It follows that 
```{math}
:name: plusminusloss

0=\sum_{i=1}^k\sum_{(x,y)\in S}\alpha_iyh_i(x) = \sum_{(x,1)\in S}\sum_{i=1}^k\alpha_ih_i(x)-\sum_{(x,-1)\in S}\sum_{i=1}^k\alpha_ih_i(x) 
\rlap{\qquad(A)}
```

First, let us assume that 

$$
\sum_{(x,-1)\in S}\sum_{i=1}^k\alpha_ih_i(x)\geq 0
$$

That means there is some $(x,-1)\in S$ that gets misclassified as class 1 by $A$, which means that $L_S(A)>0$. 
(We assume that $\sgn 0=1$.)

If, on the other hand,

$$
\sum_{(x,-1)\in S}\sum_{i=1}^k\alpha_ih_i(x)<0
$$

then [(A)](plusminusloss) implies

$$
\sum_{(x,1)\in S}\sum_{i=1}^k\alpha_ih_i(x)<0,
$$

which means that some $(x,1)\in S$ is misclassified by $A$ as having class -1. Again, this results in a 
positive training error. 
````


```{prf:Lemma}
:label: fiftytwolem
:nonumber: 

There is a set $S\subseteq \{0,1\}^{n+1}\times \{-1,1\}$ such that 
for every decision tree $T$ with at most $n$ inner nodes it holds that 

$$
L_S(T)=\tfrac{1}{2}
$$
```

````{prf:Proof}
Define 

$$
f:\{0,1\}^{n+1}\to\{0,1\},\quad x\mapsto \begin{cases}
1& \text{ if }\sum_{i=1}^{n+1}\text{ even}\\
-1& \text{otherwise}
\end{cases}
$$

and 

$$
S=\{ (x,f(x)): x\in\{0,1\}^{n+1}\}
$$

Consider a leaf of the decision tree. 
Each decision on the path to the leaf will
restrict one coordinate of the data vectors $x\in\{0,1\}^{n+1}$,
ie, there is a set $J\subseteq \{1,\ldots, n+1\}$ 
and a function $\tau: J\to \{0,1\}$ s.t.
the subset $L$ of $\{0,1\}^{n+1}$ that is classified by the leaf is

$$
L=\{x\in\{0,1\}^{n+1}: x_j=\tau(j)\text{ for all }j\in J\}
$$

In particular, if there is an $i\in \{1,\ldots, n+1\}\setminus J$ then all the points 
in $L$ come in pairs $x,x'$, where $x$ and $x'$ are identical 
in all coordinates except for coordinate $i$. Then, one of $x,x'$
has class 1, and the other -1. As $T$ attributes
the same class to every datapoint in $L$, we'd obtain an 
error rate of 50\%. 

So, is there always such an $i$? Yes, as $|J|$ is bounded by
the depth of the tree, which in turn is bounded by the number
of inner nodes of $T$. 
````

```{prf:Theorem}
:nonumber:

Let $\mathcal W_n$ denote the set of decision trees with $n$ inner nodes. Then there is a Boolean function
$f:\{0,1\}^{n+1}\to\{0,1\}$ s.t.

$$
\mathcal A_n=\{\sgn\left(\sum_{i=1}^k\alpha_ih_i: \alpha_i\in\mathbb R,\, h_i\in\mathcal W\right) \}
$$

cannot realise $f$. 
```

```{prf:Proof}
Let $S$ be the set from the second {prf:ref}`fiftytwolem`, and define
$f$ as 

$$
f(x)=\begin{cases} 1 &\text{ if }(x,1)\in S\\
0 & \text{ otherwise}
\end{cases}
$$

Then, by the first {prf:ref}`fiftylem`, every classifier in $\mathcal A_n$ will have positive
training error, ie, will not realise $f$.
```


It should be noted, though, that a Boolean training set is a particularly tough set for 
AdaBoost as axis-parallel decision rules cannot isolate datapoints. 

In contrast, if the data 
of some training set $S$ has all distinct entries along some axis then an additive model
such as AdaBoost can, in principle, perfectly fit $S$, even if the base classifiers $\mathcal W$
are mere decision stumps. (This is because two decision stumps can work together
to select for all datapoints $x$ with $x_i\in[a,b]$.)


