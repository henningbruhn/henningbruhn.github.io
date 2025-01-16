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

Properties of neural networks
=============================


ReLU networks and piecewise affine functions
--------------------------------------------

We've trained and studied neural networks -- but what kind of functions do they actually encode?
It turns out, ReLU networks encode quite simple functions, namely piecewise affine functions.

% THIS IS FROM Hein, Andriushchenko, Bitterwolf (2019)

A function $f:\mathbb R^n\to\mathbb R^m$ is *piecewise affine* if there 
are finitely many polyhedra $\ph Q_1,\ldots,\ph Q_s\subseteq\mathbb R^n$
such that $\mathbb R^n=\bigcup_{i=1}^s\ph Q_i$ and $f|_{\ph Q_i}$
is an affine function for every $i=1,\ldots, s$. The 
polyhedra $\ph Q_i$ are the *pieces* of $f$.
The smallest number of pieces $\ph Q_1,\ldots,\ph Q_s$
such that $f|_{\ph Q_i}$ is affine for every $i=1,\ldots,s$
is the *piece-number* of $f$.

```{prf:Theorem}
The function computed by 
a (leaky) ReLU neural network with linear (or ReLU, or leaky ReLU) output layer
is a continuous piecewise affine function.
```

````{prf:Proof}
The function $f$ computed by 
a (leaky) ReLU neural network with linear output layer of depth $k$ can be written as the concatenation 
of affine functions $L_1,\ldots, L_k$:

$$
f=L_k\circ \sigma \circ L_{k-1} \circ \ldots \circ \sigma \circ L_1,
$$

where $\sigma$ is the (leaky) ReLU function. We immediately see that $f$ is continuous as it 
is a concatenation of continuous functions. 

We do induction on the depth $k$ of the network. For $k=1$, the network is simply an affine function.
So, consider a depth $k>1$. Then 

$$
g=L_{k-1} \circ \sigma\circ \ldots \circ \sigma \circ L_1
$$

is piecewise affine with pieces $\ph Q_1,\ldots, \ph Q_M$. 

Consider a piece $\ph Q_i$. Restricted to $\ph Q_i$ the function $g:\mathbb R^n\to\mathbb R^m$ can be expressed 
as $x\mapsto Ax+b$ for some matrix $A$ and vector $b$. 
For each subset $S\subseteq \{1,\ldots,m\}$ 
we split $\ph Q_i$ into parts $\ph Q^{(S)}_i$ defined as 

$$
\ph Q_i^{(S)}=\{x\in\ph Q_i:(Ax+b)_s\geq 0\text{ for }s\in S\text{ and }
(Ax+b)_t\leq 0\text{ for }t\notin S\}
$$

Note that each $Q_i^{(S)}$ is a polyhedron since $\ph Q_i$ is one.
Moreover, $\sigma\circ g$ restricted to $\ph Q_i^{(S)}$ is an affine 
function, and then so is $f=L_k\circ\sigma\circ g$. Thus we see that $f$ is piecewise 
affine with pieces $\ph Q_i^{(S)}$.
````

How many pieces does the function computed by a neural network have? Let's find out. 

````{prf:Lemma}
:label: pacomplem

Let $f,g:\mathbb R\to\mathbb R$ be piecewise affine functions, and let
$f$ have piece-number $k$, and let $g$ have piece-number $\ell$. Then

1. $f+h$ has piece-number at most $k+\ell$.
2. $f\circ g$ has piece-number at most $k\ell$.
````

````{prf:Proof}
1. Consider the endpoints of the $k$ pieces of $f$ and $\ell$ pieces of $g$, 
order their union $p_1<p_2<\ldots < p_s$  by size. As $f$ has $k-1$ such points, and $g$ has $\ell-1$ such points,
it follows that $s\leq k+\ell-2$. (Why the $-1$? Because the first piece and the last piece
of $f$ and of $g$ are infinite intervals.) Then $f+g$ is affine on $[p_i,p_{i+1}]$
for each $i=0,\ldots, s$, where we put $p_0=-\infty$ and $p_{s+1}=\infty$.

2. Let $J$ be one the $\ell$ pieces of $g$, and let $I_1,\ldots, I_k$ be the 
$k$ pieces of $f$. Then $g|J$ is an affine function, and thus

$$
J\cap g^{-1}(I_1),\ldots, J\cap g^{-1}(I_k)
$$

is a partition of $J$ into at most $k$ intervals. On each of these $f\circ g$ is affine.
In total there are at most $k\ell$ such intervals.
````

````{prf:Theorem}
:label: nnpiecesthm
Let $\mathcal N$ be a (leaky) ReLU network with one input, one output node and $L-1$ hidden layers such that 
layer $\ell$ has $n_\ell$ neurons. Set $\overline n=\sum_{\ell=1}^Ln_\ell$.
Then $\mathcal N$ compute a piecewise affine function with 
piece-number at most 

$$
2^L\prod_{\ell=1}^Ln_\ell \leq \left(\frac{2\overline n}{L}\right)^L
$$

````

````{prf:Proof}
We proceed by induction over the number of layers. The induction start, with $L=0$, is trivial. 
For the induction step, assume the result to be proved for $L-1$. 

Then, each of the $n_\ell$ neurons in the penultimate layer computes a piecewise affine function
with at most 

$$
2^{L-1}\prod_{\ell=1}^{L-1}n_\ell 
$$
 
pieces. By {prf:ref}`pacomplem` 2., this implies that

$$
g^{(L)}:x\mapsto W^{(L)}f^{(L-1)}(x)+b^{(L)}
$$

is a piecewise affine function with at most 

$$
n_L\cdot 2^{L-1}\prod_{\ell=1}^{L-1}n_\ell 
$$

many pieces. Next, we observe that the leaky ReLU function $\sigma$ is piecewise affine with two pieces. Thus, 
{prf:ref}`pacomplem` 2. yields that $f^{(L)}(x)=\sigma(g^{(L)}(x))$ is piecewise affine with at most

$$
2n_L\cdot 2^{L-1}\prod_{\ell=1}^{L-1}n_\ell = 2^L\prod_{\ell=1}^Ln_\ell
$$
pieces.

Finally, the upper bound in the statement  follows directly from the inequality of the arithmetic and the 
geometric mean.
````

```{prf:Theorem}
:label: CPWLthm
A function $f:\mathbb R^n\to\mathbb R$ is continuous piecewise affine if and only
if there are affine functions $g_i,h_i$, $i=1,\ldots,N$ such that 

$$
f(x)=\max_{i=1,\ldots,N}g_i(x)-\max_{j=1,\ldots,N}h_j(x)\text{ for all }x\in\mathbb R^n
$$
 
```

```{figure} pix/pw_aff.png
:name: pwafffig
:width: 12cm

An illustration of {prf:ref}`CPWLthm`: The piecewise affine function $f$
can be expressed as the sum of the concave function $g$ and the convex function $h$. The 
convex function $h$ is the pointwise max of two affine functions (dashed), while the concave
function $h$ is the min of two affine functions (dashed). Note that $-g$ is a convex
function and that $-g$ can then be seen as the max of two affine functions.
```

````{prf:Proof}
We will only do the proof of the $\Leftarrow$-direction. The other direction
follows, for instance, from a result of Wang and Sun (2005).

First, we note that if $g,h$ are continuous piecewise affine functions then 
this is also the case for $g-h$. Thus, by
induction, it suffices to prove that:
 if $g_1,g_2:\mathbb R^n\to\mathbb R$
are continuous piecewise affine, then $g:x\mapsto \max\{g_1(x),g_2(x)\}$ is 
continuous and piecewise affine, too. That $g$ is contiuous is elementary -- we only
prove that $g$ is piecewise affine. 

For this, let $\ph Q_1^1,\ldots, \ph Q_k^1$ be the pieces of $g_1$, and let 
$\ph Q_1^2,\ldots, \ph Q_\ell^2$ be the ones of $g_2$. We first note that each $\ph Q_i^1\cap \ph Q_j^2$
is again a polyhedron (this is a simple fact about polyhedra), and that each of $g_1$ and 
$g_2$ is affine on $\ph P'_{ij}=\ph Q_i^1\cap \ph Q_j^2$. We split each $\ph P'_{ij}$
into two parts:

$$
\ph P_{ij1}=\{x\in \ph P'_{ij}: g_1(x)\geq g_2(x)\}\text { and }
\ph P_{ij2}=\{x\in \ph P'_{ij}: g_1(x)\leq g_2(x)\}
$$
 
As $g_1,g_2$ are affine on $P'_{ij}$, both $\ph P_{ij1}$ and $\ph P_{ij2}$ are polyhedra. 
(Again, this is a basic fact about polyhedra.) As $f|{\ph P_{ij1}}=g_1|{\ph P_{ij1}}$
and $f|{\ph P_{ij2}}=g_2|{\ph P_{ij2}}$, it follows that $f$ restricted to each $\ph P_{ijr}$
is affine, which proves $f$ to be piecewise affine.
````

The number $N$ may be seen as a measure of how complex the piecewise affine function $f$ is. 
It would be nice to put $N$ in relation with the number of pieces (and perhaps the dimension $n$)
but at the moment I do not see a good way how to do that.
Note, moreover, that each of $\max_{i=1,\ldots,N}g_i(x)$ and $\max_{j=1,\ldots,N}h_j(x)$
is a convex function. (Recall {prf:ref}`suplem`.)


Universal approximators
-----------------------

How expressive are neural networks? We have already seen above that every Boolean function 
can be realised as a ReLU network. What about more complicated functions? Since ReLU networks
encode piecewise affine functions such a network cannot reproduce any function that is not 
piecewise affine -- but can every piecewise affine function be realised? Yes!

Let's define the *depth of a neural network* as the number of its layers not counting the input
layer. For example, a shallow neural network with input layer, one hidden layer and output layer
has depth 2. Let's say that the neural network has *width* at most $d$ if no layer,
except possibly for the input layer,
has more than $d$ neurons.[^Han17] 

````{prf:Theorem} Hanin (2017)
:label: Haninthm

Let $f:\mathbb R^n\to\mathbb R$ be a continuous piecewise affine function.
If 

$$
f(x)=\max_{i=1,\ldots,N}g_i(x)-\max_{j=1,\ldots,N}h_j(x)\text{ for all }x\in\mathbb R^n,
$$
 
where the $g_i$ and $h_j$ are affine functions,
then exists a ReLU neural network $F$ with linear output layer of depth at most $3N$
and width $4n+12$ such that $F(x)=f(x)$ for all $x\in\mathbb R^n$.
````

[^Han17]: *Universal function approximation by deep neural nets with bounded width
and ReLU activation*, B. Hanin (2017), [arXiv:1708.02691](https://arxiv.org/abs/1708.02691)

What is remarkable about this theorem? 
A thin but deep network can express every function that any ReLU network, of whatever width, 
can ever compute, and modern neural networks are precisely of that type: relatively thin but with many layers. 

For the following lemmas, I will write *ReLU network* to mean 
*ReLU neural network with linear output layer*.
Consider a neural network of the form

$$
L^{(k)}\circ\sigma^{(k-1)}\circ L^{(k-1)}\circ \ldots L^{(1)}
$$

where each $L^{(i)}$ is an affine function, and each $\sigma^{(i)}:\mathbb R^{n_i}\to\mathbb R^{n_i}$
is the activation function of layer $i$. For this section, let us call $\sigma^{(i)}$
a \ndefi{linear/ReLU} activation function if each entry of the vector function 
is the identity or the ReLU function. If each activation function is linear/ReLU 
(except for the output, where we require no activation) then we call the network
a *linear/ReLU network*. (Yes, that is clunky, but we'll only use the name here.)

We can always get rid of linear activation in hidden layers without 
increasing the size of the neural network too much.

```{prf:Lemma}
:label: replacelem
Let $\mathcal N$ be a linear/ReLU network of depth $d$ and width $w$. 
Then there is a ReLU network $\mathcal N'$ of depth $d$ and width at most $2w$
 that computes the same function.
```

````{prf:Proof}
Consider a neuron of $\mathcal N$, not in the output layer, with linear activation.
Then the neuron computes an affine function $x\mapsto \trsp wx+\beta$, where $w$ is a vector
of $\beta$ a number. Then

$$
x\mapsto \text{ReLU}(\trsp wx+\beta)-\text{ReLU}(-\trsp wx-\beta)
$$

represents the same function $x\mapsto \trsp wx+\beta$. Thus, each neuron in $\mathcal N$
in each hidden layer with linear activation can be replaced by two ReLU neurons, without 
changing the computed function. 
````

Next, we need to think about how to compose smaller networks into a larger network.
```{prf:Lemma}
:label: NNaddlem
Let $f_1,f_2:\mathbb R^n\to\mathbb R^m$ be two functions that can be computed by 
linear/ReLU networks of depth $k$ and width $\ell$. Then $f_1+f_2$ can be computed 
by a linear/ReLU network of depth $k$ and width $2\ell$.
```

```{prf:Proof}
Put the networks for $f_1,f_2$ in parallel, with input and output layer identified.
Set the weights between the two networks to $0$. 
```

```{prf:Lemma}
:label: NNcomplem
Let $f_1:\mathbb R^n\to\mathbb R^m$ be a function that can be computed by a linear/ReLU
network of depth $k_1$ and width $\ell_1$, and 
let $f_2:\mathbb R^m\to\mathbb R^d$ be a function that can be computed by a linear/ReLU
network of depth $k_2$ and width $\ell_2$. Then $f_2\circ f_1$ can be computed by a linear/ReLU
network of depth $k_1+k_2$ and width $\max\{\ell_1,\ell_2\}$.
```

```{prf:Proof}
We simply concatenate the two networks.
```

```{prf:Lemma}
:label: smallmaxlem
The function 

$$
\mathbb R^2\to\mathbb R,\,
\twovec{x_1}{x_2}\mapsto \max(x_1,x_2)
$$

can be computed by a ReLU network of depth $2$ and width $3$. 
```


````{prf:Proof}

The network...
```{image} pix/smallmax.png
:width: 8cm
:align: center
```

...realises $(x_1,x_2)\mapsto \text{ReLU}(x_1-x_2)+\text{ReLU}(x_2)-\text{ReLU}(-x_2)$.
````

```{prf:Lemma}
:label: NNmaxlem
Let $g_1,\ldots,g_N:\mathbb R^n\to\mathbb R$ be  functions that can be computed by 
linear/ReLU networks of depth $k$ and width $\ell$. Then $\max_{i=1,\ldots, N} g_i(x)$ can be computed 
by a linear/ReLU network of depth $N(k+2)$ and width $\max\{n+3,n+\ell+2\}$.
```

````{prf:Proof}
Below, let $x\in\mathbb R^n$, and let greek letters denote real numbers.
Define 

$$
g'_1:\mathbb R^n\to\mathbb R^{n+1},\,
x\mapsto\twovec{x}{g_1(x)}
$$

and for $2=1,\ldots,N$ define  

$$
g'_i:\mathbb R^{n+1}\to\mathbb R^{n+2},\,
\twovec{x}{\alpha}\mapsto \begin{pmatrix}x\\\alpha\\g_i(x)\end{pmatrix}
$$

and observe that $g'_i$ can be realised by a linear/ReLU network of depth $k$ and width $n+\ell+2$.
Define, moreover

$$
{\max}':\mathbb R^{n+2}\to\mathbb R^{n+1},\,
\begin{pmatrix}x\\\alpha\\\beta\end{pmatrix}\mapsto\twovec{x}{\max\{\alpha,\beta\}},
$$

and observe that, by {prf:ref}`smallmaxlem`, ${\max}'$ can be realised by a linear/ReLU network
of depth 2 and width $n+3$.

We also note that ${\max}'\circ g'_2\circ g_1'$ computes the function 

$$
x\mapsto (x,\max\{g_1(x),g_2(x)\}\trsp ),
$$

and that, more generally

$$
{\max}'\circ g'_N\circ{\max}'\circ \ldots {\max}'\circ g'_2\circ g_1'
$$

computes the function $x\mapsto (x,\max_{i=1,\ldots,N}g_i(x)\trsp )$.

With {prf:ref}`NNcomplem`, we deduce that this can be realised by a linear/ReLU network
of depth $N(k+2)$ and width $\max\{n+3,n+\ell+2\}$.
````

We can now finish the proof of {prf:ref}`Haninthm`:

````{prf:Proof}
Each $g_i,h_j$ is an affine function and can therefore be computed by a ReLU network
of depth 1 and width 1. (Note that we do not count the input layer for the width.)
By {prf:ref}`NNmaxlem`, each of $\max_{i=1,\ldots,N}g_i$ and $\max_{j=1,\ldots,N}h_j$
can thus be realised by a ReLU network of depth $3N$ and width $n+3$.

Applying {prf:ref}`NNaddlem` and {prf:ref}`replacelem` finishes the proof.
````

We now use {prf:ref}`Haninthm` to convince ourselves that neural networks
are *universal approximators*: they can approximate every continuous function
very well, at least on a compact set. 

````{prf:Theorem}
:label: univthm
Let $f:\mathbb R^n\to\mathbb R$ be a continuous function, and let $C\subseteq\mathbb R^n$ be compact.
Then, for every $\epsilon>0$, there is a ReLU neural network with linear output layer that computes
a function $F:\mathbb R^n\to\mathbb R$ such that 

$$
\max_{x\in C}|f(x)-F(x)|<\epsilon
$$

````

The theorem is a direct consequence of {prf:ref}`Haninthm` and the theorem of Stone-Weierstrass:

```{prf:Theorem} Stone-Weierstrass
Let $f:\mathbb R^n\to\mathbb R$ be a continuous function, and let $C\subseteq\mathbb R^n$ be compact.
Then for every $\epsilon>0$ there is a continuous piecewise affine function $F$
such that

$$
\max_{x\in C}|f(x)-F(x)|<\epsilon
$$

```

There are also versions for non-continuous functions. Moreover, 
deep neural ReLU networks do not need to have very large width (size of layers)
to approximate a (non-crazy) function:[^LuPu17]
```{prf:Theorem} Lu et al. 
:label: univ2thm
Let $f:\mathbb R^n\to\mathbb R$ be a Lebesque-measurable function and let $\epsilon>0$. Then there is a
fully connected ReLU network of maximum width at most $n+4$ (but possibly large depth) such that 
the function $N$ represented by the network achieves

$$
\int_{\mathbb R^n}|f(x)-N(x)|dx\leq \epsilon
$$
  
```

[^LuPu17]: *The Expressive Power of Neural Networks: A View
from the Width*, Zh. Lu, H. Pu, F. Wang, Zh. Hu and L. Wan (2017)

%also: deep vs shallow. See Hanin et al: have exponential blow-up in neuron number between deep and shallow


What can we deduce from {prf:ref}`univthm` and {prf:ref}`univ2thm`? They are good news: for any classification
or regression task that is based on some non-crazy function it is possible to devise a neural network 
with very small approximation error. 

We should, however, not get overly excited about these insights. The theorems make no statement about the size
of the network, and it's also unclear how hard it is to train the neural network. That is, it could well be 
that even some easy tasks need very large networks that are virtually impossible to train. Fortunately, 
this does not seem to be the case. 

That neural networks are universal approximators is known for a long time now. The classic results, however, 
were about shallow but wide neural networks. The results above, in contrast, show that also narrow but deep networks
achieve the same. In fact, deep networks seem to be more powerful. A first indication comes from 
{prf:ref}`vcnetthm` (or rather the matching lower bound): 
the VC-dimension of neural networks increases with the number of layers. 


The saw tooth function
----------------------

Deeper neural networks are more powerful than shallow networks with the same number 
of parameters. An easy example of this is the *saw tooth function*: it can be 
computed by a deep network with only a few neurons; any shallow network, however, 
will need a very large number of neurons.[^Telnotes]

[^Telnotes]: I am following here [lecture notes](https://mjt.cs.illinois.edu/dlt/) of Telgarsky, who also proved 
the main result in this section. 

What is the saw tooth function? It is the iteration of the simple function

$$ 
\Delta:\mathbb R\to\mathbb R,\quad x\mapsto 
\begin{cases}
2x & \text{ if }x\in[0,\tfrac{1}{2}]\\
2-2x & \text{ if }x\in[\tfrac{1}{2},1]\\
0 & \text{ if }x\notin[0,1]
\end{cases}
$$
  
The function $\Delta$ is obviously piecewise affine with four pieces. It is, moreover, symmetric around $\tfrac{1}{2}$, 
ie, $\Delta(x)=\Delta(1-x)$ for all $x\in\mathbb R$.

```{figure} pix/sawtooth.png
:name: sawfig
:width: 12cm

$\Delta$, on the left. On the right: $\Delta$ iterated three times.
```


Iterating $\Delta$ results in a saw tooth pattern; see {numref}`sawfig`.
Let us prove that:

```{prf:Lemma}
:label: sawtoothlem

For every integer $\ell\geq 1$ it holds that

$$ 
\Delta^\ell(x)=\begin{cases}
2^\ell x-k+1 & \text{ if }x\in\left[(k-1)\tfrac{1}{2^\ell},k\tfrac{1}{2^\ell}\right]\text{ for odd }k\in\{1,\ldots,2^\ell\}\\
-2^\ell x+k+1 & \text{ if }x\in\left[(k-1)\tfrac{1}{2^\ell},k\tfrac{1}{2^\ell}\right]\text{ for even }k\in\{1,\ldots,2^\ell\}\\
0 & \text{ if }x\notin[0,1]
\end{cases}
$$
 
```

````{prf:Proof}
We proceed by induction on $\ell$. Assume that the lemma is already verified for $\ell-1$. Then, 
as $\Delta$ is symmetric around $\tfrac{1}{2}$, it suffices to consider an $x\in[0,\tfrac{1}{2}]$. 

Furthermore, let $k\in\{1,\ldots,2^\ell\}$ be such that $x\in\left[(k-1)\tfrac{1}{2^\ell},k\tfrac{1}{2^\ell}\right]$.
As $x\leq\tfrac{1}{2}$, it actually follows that $k\leq 2^{\ell-1}$. 
\begin{align*}
\Delta^\ell(x) & = \Delta^{\ell-1}(\Delta(x)) = \Delta^{\ell-1}(2x) 
\end{align*}
Note that $2x$ lies in 

$$ 
\left[(k-1)\tfrac{1}{2^{\ell-1}},k\tfrac{1}{2^{\ell-1}}\right],
$$
 
which means that we can apply the induction hypothesis to $2x$:
\begin{align*}
\Delta^\ell(x) & =  \Delta^{\ell-1}(2x)  = \begin{cases}
2^{\ell-1}\cdot 2x-k+1 & \text{ if $k$ odd }\\
-2^{\ell-1}\cdot 2x+k+1 & \text{ if $k$ even }
\end{cases}
\end{align*}
That is precisely what we needed to prove.
````

We immediately note an important consequence:

```{prf:Lemma}
:label: sawpieceslem
The saw tooth function $\Delta^\ell$ is a piecewise affine function with piece-number $2^\ell+2$.
```

It is easy to realise the saw tooth function with a neural network. In fact
```{math}
:label: sawrelu
\Delta(x) = 2(\text{ReLU}(x)-2\text{ReLU}(x-\tfrac{1}{2}) + \text{ReLU}(x-1))
```
This follows from a straightforward check.
Indeed, if $x\leq 0$ then 

$$
2(\text{ReLU}(x)-2\text{ReLU}(x-\tfrac{1}{2}) + \text{ReLU}(x-1)) = 2(0-2\cdot 0+0) = 0
$$
 
If $x\geq 1$ then 

$$
2(\text{ReLU}(x)-2\text{ReLU}(x-\tfrac{1}{2}) + \text{ReLU}(x-1)) = 2(x-2(x-\tfrac{1}{2})+x-1) = 0
$$
 
and so on.

As a consequence, $\Delta$ can be expressed as a small ReLU network like this:
```{image} pix/sawnet.png
:alt: one
:width: 6cm
:align: center
```
(Here the weights are drawn on the edges, while the biases are put close to their neuron.)

By concatenating these nets we get:
```{prf:Lemma}
For every integer $\ell\geq 1$, the function $\Delta^\ell$ can be realised by a ReLU network
with $2\ell$ layers and $4\ell$ neurons.
```
(I do not count the input node.)

````{prf:Proof}
$\,$
```{image} pix/sawnet2.png
:alt: one
:width: 12cm
:align: center
```

(Yes, with a bit more care, we can get rid of $\ell-1$ neurons.)
````

We can observe that deeper networks are indeed more powerful than shallow ones.[^Tel16]
````{prf:Theorem} Telgarsky

Let $L\geq 2$ be an integer, and set $\ell=L^2+4$. Then

1. There is a ReLU network with $2L^2+8$ layers and at most $4L^2+16$ neurons that computes the saw tooth function $\Delta^{\ell}$. 

2. Let $\mathcal N$ be a ReLU network with at most $L$ layers and at most $2^L$ neurons. Then

$$
\int_{[0,1]} |\mathcal N(x)-\Delta^{\ell}(x)|dx \geq \tfrac{1}{32}
$$

````

[^Tel16]: *Benefits of depth in neural networks*, M. Telgarsky (2016), [arXiv:1602.04485](https://arxiv.org/abs/1602.04485)

As a consequence, any ReLU network with at most $L$ layers that computes $\Delta^{L^2+4}$ needs to have at least $2^L$ neurons --
quite a lot more than the $\bigO(L^2)$ neurons a deeper network needs. 

````{prf:Proof}
1\. This is a direct consequence of {eq}`sawrelu`.

2\.
Consider the saw tooth function $\Delta^\ell$, and draw a horizontal line at height $y=\tfrac{1}{2}$. 
Between that line and the saw tooth function there are $2^{\ell}-1$ small triangles of area $\tfrac{1}{2^{\ell+2}}$, 
shown in grey here:

```{image} pix/sawtooth2.png
:alt: one
:width: 12cm
:align: center
```

We denote the set of these triangles by $\mathcal T$. 
We note for later:
```{math}
:label: numtrigs
|\mathcal T|=2^\ell-1=2^{L^2+4}-1>2^{L^2+3}
```

Let $f:\mathbb R\to\mathbb R$ be some function (perhaps $\mathcal N$, the function computed by the neural network),
and consider one of the triangles $T\in\mathcal T$,
and denote by $I$ the interval of $x$-values, where $T$ intersects the $\tfrac{1}{2}$-line (its shadow on the $x$-axis).
We say that $f$ *misses* $T$ if $T$ lies above the $\tfrac{1}{2}$-line and if $f|I\leq \tfrac{1}{2}$, or if $T$ lies
below the $\tfrac{1}{2}$-line and $f|I\geq \tfrac{1}{2}$. Otherwise, we say that $f$ *hits* $T$.
Triangles missed by $\mathcal N$ yield a lower bound as follows:

$$
\int_{[0,1]}|\mathcal N(x)-\Delta^{\ell}(x)|dx \geq \#\text{missed triangles}\cdot\text{triangle area}
$$

We have computed the triangle area. To get an estimate for the number of missed triangles, 
we will upper-bound the number of triangles that are hit by $\mathcal N$. 

```{figure} pix/sawtooth3.png
:name: fig:sawtrigs
:width: 15cm

Triangles missed by second affine function in grey. Function realised by $\mathcal N$ in red.
```

By {prf:ref}`nnpiecesthm`, the neural network $\mathcal N$ computes a piecewise
affine function with piece-number at most 

$$
\left(\frac{2\cdot 2^L}{L}\right)^L\leq 2^{L^2}
$$
 
Let the number of pieces be $p\leq 2^{L^2}$, and let
 $I_1,\ldots, I_p$ be the pieces.
Consider a piece $I_j$, and denote by $\mathcal T_j$ those triangles in $\mathcal T$
that have their leftmost vertex in $I_j$. To make all the sets $\mathcal T_i,\mathcal T_j$
pairwise disjoint, we shrink $I_1,\ldots, I_t$ to disjoint, half-open intervals that partition $\mathbb R$.
Then $\sum_{j=1}^p|\mathcal T_j|=|\mathcal T|$. 

How many of the triangles in $\mathcal T_j$ are hit by the affine function $f:=\mathcal N|_{I_j}$? 
Say $f$ hits $r$ of the triangles that lie above the $\tfrac{1}{2}$-line, then it will miss 
at least $r-1$ triangles below that line. Likewise, if $f$ hits $s$ triangles below the line
it will miss at least $s-1$ triangles above. Thus $|\mathcal T_j|\geq 2r-1+2s-1$, 
and the number of triangles hit by $f$ is 

$$
r+s =\frac{2(r+s)-2}{2}+1\leq \frac{|\mathcal T_j|}{2}+1
$$
 

At the right end of $I_j$ there may be a triangle in $\mathcal T_j$ that is not hit by $f$ but
by the subsequent affine function, ie, is hit by $\mathcal N$ but not by $f$.  
We will be genereous and simple assume that, in the worst, case
all these triangles are hit. That is, if $h_j$ is the number of triangle in $\mathcal T_j$ hit by $\mathcal N$ 
then 

$$
h_j\leq \frac{|\mathcal T_j|}{2}+2
$$

As a consequence, we get that the total number of triangles hit by $\mathcal N$ is 

$$
\sum_{j=1}^p h_j \leq \sum_{j=1}^p\left(\frac{|\mathcal T_j|}{2}+2\right)=\frac{|\mathcal T|}{2}+2p
$$


Thus, the number of triangles missed by $\mathcal N$ is at least:
\begin{align*}
\#\text{missed triangles} &\geq |\mathcal T|-\left(\frac{|\mathcal T|}{2}+2p\right)
= \frac{|\mathcal T|}{2}-2p \\
&\geq \frac{2^{L^2+3}}{2}-2\cdot 2^{L^2} = 2^{L^2+2}-2^{L^2+1} = 2^{L^2+1},
\end{align*}
where we have used {eq}`numtrigs` and that $p\leq 2^{L^2}$.

We finish with
\begin{align*}
\int_{[0,1]}|\mathcal N(x)-\Delta^{\ell}(x)|dx & \geq \#\text{missed triangles}\cdot\text{triangle area}\\
& \geq 2^{L^2+1}\cdot \tfrac{1}{2^{L^2+6}} = \frac{1}{32}
\end{align*}
````

Neural networks are sometimes overconfident
-------------------------------------------

[^foolcode]
In practice it is sometimes observed that neural networks confidently 
classify some image data, with confidence levels approaching 100\%, 
even though the input data is just white noise. That is, in MNIST for 
example, it is possible to generate noise pictures that a neural network
will happily claim show a '7', and that with 99.8\% certainty.

[^foolcode]: {-} [{{ codeicon }}fool](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/neural_networks/fool.ipynb)

```{figure} pix/fool.png
:name: foolfig
:width: 12cm

A neural network was trained on the MNIST data set. It recognises the '3' on the left correctly, 
and with high confidence. It also recognises noise and the picture of a shoe as a '3', again with high confidence.
```


Hein et al. (2019)[^Hein19]
offer an explanation, why this is not surprising and actually should be 
expected in many applications.

[^Hein19]: *Why ReLU networks yield high-confidence predictions far away from
the training data and how to mitigate the problem*, M. Hein, M. Andriushchenko and J. Bitterwolf (2019), [arXiv:1812.05720](https://arxiv.org/abs/1812.05720)

```{prf:Lemma}
:label: raylem

Let  $f:\mathbb R^n\to\mathbb R^m$ be a  piecewise affine function
with pieces $\ph Q_1,\ldots, \ph Q_s$. 
For every $x\in\mathbb R^n$ there is an $\alpha\geq 1$ and a 
piece $\ph Q_t$
such that $\beta x\in\ph Q_t$ for all $\beta\geq\alpha$.
```

````{prf:Proof}
Put $R=\{\gamma x:\gamma\geq 1\}$. For every $\ph Q_i$ that 
meets $R$ pick some $\alpha_i\geq 1$ such that $\alpha_i x\in\ph Q_i$.

Suppose that the conclusion of the lemma is false. 
Then, for every $vQ_i$ that meets $R$ there must be some $\beta_i>\alpha_i$
with $\beta_i x\notin \ph Q_i$. Let $\beta^*=\max_i\beta_i$. 

The point $\beta^*x$ must lie in some $\ph Q_j$. Thus $\ph Q_j$ meets $R$, and $\alpha_j$ is defined.
In particular, $\alpha_j<\beta_j<\beta^*$. Then, however, $\alpha_jx,\beta^*x\in \ph Q_j$
but $\beta_jx\notin \ph Q_j$, which contradicts that $\ph Q_j$ is convex. 
````

```{prf:Theorem}
:label: foolthm
Let $N$ be  a ReLU network with  softmax output layer, and
let $f:\mathbb R^n\to\mathbb R^K$ be the piecewise affine function 
such that $N$ computes the function $\softmax\circ f$, where $\softmax$ is the 
softmax function.
Let $\ph Q_1,\ldots, \ph Q_s$ be the pieces of $f$, and let 
$f|_{\ph Q_t}$ be described by the affine function $x\mapsto A^{(t)}x+b^{(t)}$.
If, for all $t=1,\ldots, s$, the matrix $A^{(t)}$ does not have identical 
rows then for almost every $x\in\mathbb R^n$ there exists 
a class $k\in\{1,\ldots,K\}$ such that 
the confidence of $N$ that $\alpha x$ lies in $k$ tends to 1
as $\alpha\to\infty$, ie

$$
\lim_{\alpha\to\infty}\frac{e^{f_k(\alpha x)}}{\sum_{\ell=1}^Ke^{f_\ell(\alpha x)}} = 1
$$

```

Here, *almost every* means *except for a subset of Lebesgue measure 0*.

Now, the theorem makes a good number of assumptions. 
Some are uncritical: indeed, we know that a ReLU network minus its softmax output
layer simply computes a piecewise affine function. What about the assumption
on the affine functions $x\mapsto A^{(t)}x+b^{(t)}$ on the pieces? 
If two rows of $A^{(t)}$ are identical that means that
on the whole corresponding piece $\ph Q_t$, the network cannot effectively 
distinguish between the classes associated with the duplicated rows: the 
probability assigned to one class will be a multiple of the probability
assigned to the other class -- over the whole piece. This seems a very special 
situation and, according to Hein et al., is practically never observed.

Why should we interpret the classification of $\alpha x$, with $\alpha$ becoming
larger and larger, as bogus? First, $x$ can be taken to contain just noise, 
ie, random values. Thus, $\alpha x$ will just be noise amplified to 
very large values: clearly, not a cat picture.

Let's do the proof of {prf:ref}`foolthm`.
````{prf:Proof}
For every $t$ and distinct row numbers $i,j$ the set of all $x\in\mathbb R^n$ 
with $A^{(t)}_{i,\bullet} x=A^{(t)}_{j,\bullet}x$ has Lebesgue measure 0. 
Since there are only finitely many such triples $t,i,j$ we deduce
that, except on a null set $\mathcal N$, 
we always have $A^{(t)}_{i,\bullet} x\neq A^{(t)}_{j,\bullet}x$.

Now consider some $x\in\mathbb R^n\setminus\mathcal N$ and apply {prf:ref}`raylem`
to $x$ and $f$. Thus, there is some $\alpha^*\geq 1$ and $t$ such that 
$\beta x\in\ph Q_t$ for all $\beta\geq \alpha^*$. Since $x\notin\mathcal N$
there is a row number $k$ such that 

$$
A^{(t)}_{k,\bullet} x>A^{(t)}_{i,\bullet} x\text{ for all }i\neq k
$$

Thus

$$
A^{(t)}_{i,\bullet} (\beta x)+b^{(t)}_i-(A^{(t)}_{k,\bullet} (\beta x)+b^{(t)}_k)\to -\infty
\text{ as }\beta\to\infty
$$

As a consequence we get 
\begin{align*}
\lim_{\alpha\to\infty}\frac{e^{f_k(\alpha x)}}{\sum_{\ell=1}^Ke^{f_\ell(\alpha x)}} & = 
\lim_{\alpha\to\infty}\frac{e^{A^{(t)}_{k,\bullet} \alpha x+b^{(t)}_k}}
{ \sum_{\ell=1}^Ke^{ A^{(t)}_{\ell,\bullet} \alpha x+b^{(t)}_\ell} } \\
& =  
\lim_{\alpha\to\infty}\frac{1}
{1+ \sum_{\ell\neq k}e^{ A^{(t)}_{\ell,\bullet} \alpha x+b^{(t)}_\ell
-(A^{(t)}_{k,\bullet} \alpha x+b^{(t)}_k)} } \to 1,
\end{align*}
as the exponent in the exponential function tends to $-\infty$.
````

So, how well does the theorem explain overconfident predictions? 
The theorem (and its proof) hinges on the fact that we can move $x$ away 
from the origin as far as we like: the theorem makes a statement 
on $\alpha x$ with large $\alpha$. In practically all applications, 
the data is constrained to limited values: pixels have grey value in $[0,1]$, 
the weight of humans is never larger than 500kg (and never negative), 
and even the wealthiest human being doesn't have a net worth measuring in the trillions.

Often, however, overconfident predictions also occur for data 
that is subject to the same constraints as legit input data; see 
Nguyen et al. (2015).[^Ngu15]

[^Ngu15]: *Deep Neural Networks are Easily Fooled:
High Confidence Predictions for Unrecognizable Images*, 
A. Nguyen, J. Yosinski and J. Clune (2015), [arXiv:1412.1897](https://arxiv.org/abs/1412.1897)



