$\newcommand{\bigO}{O}
\newcommand{\trsp}[1]{#1^\intercal} % transpose
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\DeclareMathOperator*{\vari}{\mathbb{V}}   % Probability
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
$


Maximum inner product search
============================

% see https://en.wikipedia.org/wiki/Vector_database
*Vector databases* are becoming more and more important. 
What's a vector database? A system to store a large number of vectors 
$x^{(1)},\ldots, x^{(n)}\in\mathbb R^d$
in such a way that a (approximate) nearest neighbour search can be performed efficiently.    
A recommender system, for instance, might
store the preferences of the users encoded as vectors; for a new user the five most similar
known users could be computed in order to recommend the products or services they prefered. 
Another application comes from word or document embeddings: A number of vector representation 
of documents are stored in the database; a user may then formulate a query ("which Tom Stoppard play 
features Hamlet as a side character?") that is transformed into a vector; the documents with 
most similar vector representation are then returned.

What *most similar* means will differ from application to application. Often it may
simply mean: the largest scalar product. That is, given a query $q\in\mathbb R^d$ we look for the $x^{(i)}$
with largest $\trsp{q}x^{(i)}$. In that case, the problem is known as *maximum inner product search* (or MIPS).

At first, the computational problem may appear to have an easy solution -- after all, scalar 
products can be computed very efficiently. With $n$ vectors in the database, each with length $d$,
checking every scalar product amounts to a running time of $\bigO(nd)$. The number of vectors, $n$, 
however, will usually be extremely large, perhaps even in the billions, and the dimension $d$ may 
have three or four digits. Moreover, queries typically need to be answered very quickly. A recommender system, 
for example, could easily need to address hundreds or thousands of queries every second. 
As a result, a running time of $\bigO(nd)$ may be too slow. 
How can this be sped up?

:::{dropdown} Retrieval augmented generation
:color: success
:icon: telescope

It is not easy to keep an AI chatbot up-to-date with current affairs. 
The large language model (LLM) it is based on is trained on a snapshot of data 
available at training time (perhaps all of wikipedia as of 07/10/24). Later
events (announcement of the Physics Nobel Prize on 08/10/24) are thus not immediately accessible 
to the bot. 

Re-training an LLM to incorporate later events is resource intensive and therefore only 
done rarely. An alternative is to provide the AI chatbot with an external memory. One such 
method is *retrieval augmented generation*, or RAG. 

In RAG, the user query is transformed into a vector via a topic embedding that is compared to 
data in a vector database (this could be a current snapshot of wikipedia). The perhaps ten documents
that are most similar to the query are located and added to the prompt of the AI chatbot. The
bot then generates the response based on the query and the search results. 

*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 
P. Lewis et al. (2021), [arXiv:2005.11401](https://arxiv.org/pdf/2005.11401) 
:::

Vector quantisation
-------------------

% this is largely from
% Quantization based Fast Inner Product Search
% Ruiqi Guo, Sanjiv Kumar, Krzysztof Choromanski, David Simcha
% arXiv:1509.01469
Let $x^{(1)},\ldots x^{(n)}\in\mathbb R^d$ be the vectors  that make up the database,
let $k,m>0$ be integers, and let $\ell=\tfrac{d}{m}$, which we assume to be an integer.[^quant]

[^quant]: *Quantization based Fast Inner Product Search*, R. Guo, S. Kumar, K. Choromanski and D. Simcha (2015), [arXiv:1509.01469](https://arxiv.org/abs/1509.01469)

We split each vector $x\in\mathbb R^d$ into $m$ vectors each of length $\ell$:

$$x=\begin{pmatrix}x_1\\x_2\\\vdots\\x_m\end{pmatrix}, \text{ with }x_j\in\mathbb R^\ell\text{ for }j=1,\ldots, m$$

(In practice, this partition of $\mathbb R^d$ into subspaces is achieved by random sampling of the entry dimensions.)

Then, for $j=1,\ldots,m$ we compute representative vectors $c_{1j},\ldots,c_{kj}\in\mathbb R^\ell$.
These are used to replace the database vectors by simpler vectors: For each $i=1,\ldots, n$ and $j=1,\ldots, m$ 
we find a suitable $\hat x^{(i)}_j\in\{c_{j1},\ldots,c_{jk}\}$. That is, we basically replace $x^{(i)}_j$ by one of $\{c_{j1},\ldots,c_{jk}\}$.
In this way $x^{(i)}$ is replaced by 

$$
\hat x^{(i)}=\begin{pmatrix}\hat x^{(i)}_1\\\vdots\\\hat x^{(i)}_m\end{pmatrix}
$$

For a query $q\in\mathbb R^d$ we then approximate $\trsp qx^{(i)}\approx \trsp q\hat x^{(i)}$. 

Before we look at $q\hat x^{(i)}$ let me point out that replacing each $x^{(i)}$ by $\hat x^{(i)}$ 
already results in a welcome compression of the data. Indeed, we only need to store all $c_{js}$, $j=1,\ldots, m$, $s=1,\ldots, k$
and, instead of $x^{(i)}$ we store a vector $\hat z^{(i)}\in\{1,\ldots,k\}^m$ with $\hat z^{(i)}_j=s$ if and only if $\hat x^{(i)}_j=c_{js}$.

What is the computational complexity to compute $\trsp q\hat x^{(i)}$ for all $i$?
We write

$$
\trsp q\hat x^{(i)} = \sum_{j=1}^m\trsp q_j\hat x^{(i)}_j
$$

Note that $\hat x^{(i)}_j$ is one of $c_{j1},\ldots, c_{jk}$. We first compute all scalar products $\trsp q_jc_{js}$ and 
put them in a look-up table. Computing the scalar product takes $\bigO(mk\ell)$ time as $j$ runs from $1$ to $m$, $s$ runs
from $1$ to $k$ and as the vectors have length $\ell$. With $\ell=\tfrac{d}{m}$ the running time reduces to $\bigO(kd)$.
Then, using the look-up table, we compute for all $i$ the scalar product $\trsp q\hat x^{(i)}$, with a running time of 
$\bigO(nm)$. In total we obtain a running time of $\bigO(kd+nm)$. As $n$ is typically the by far largest quantity among
$d,k,n,m$ the running time is dominated by $\bigO(nm)$, and as usually $m \ll d$, we achieve a substantial reduction 
in comparison to the running time $\bigO(nd)$ of the naive algorithm.

How should the representatives $c_{ji},\ldots,c_{jk}\in\mathbb R^\ell$ be chosen? Well, $\trsp q\hat x^{(i)}$
should be a good approximation of $\trsp qx^{(i)}$, which is the case if $\trsp q_j\hat x^{(i)}_j$ is a 
good approximation of $\trsp q_jx^{(i)}_j$ for every $j\in\{1,\ldots, m\}$. So, let's fix $j$ and 
let us assume that the queries $q$ are drawn from some distribution $\mathcal D$. 
Then, arguably, we should choose $c_{j1},\ldots,c_{jk}$ such that 

```{math}
:label: vqobj
\expec_{q\sim\mathcal D}\left[\sum_{i=1}^n(\trsp q_jx^{(i)}_j-\trsp q_j\hat x^{(i)}_j)^2\right]
```

is minimised. Let us rewrite that.

\begin{align*}
\expec_{q\sim\mathcal D}\left[\sum_{i=1}^n\left(\trsp q_jx^{(i)}_j-\trsp q_j\hat x^{(i)}_j\right)^2\right]
&= \sum_{i=1}^n\expec_{q\sim\mathcal D}\left[\left(\trsp q_j\left(x^{(i)}_j-\hat x^{(i)}_j\right)\right)^2\right]\\
&=\sum_{i=1}^n\expec_{q\sim\mathcal D}\left[\left( \cos\theta_{ij} ||q_j||\cdot ||x^{(i)}_j-\hat x^{(i)}_j|| \right)^2\right]\\
&=\sum_{i=1}^n||x^{(i)}_j-\hat x^{(i)}_j||^2\cdot\expec_{q\sim\mathcal D}\left[\left( \cos\theta_{ij} ||q_j||\right)^2\right],
\end{align*}

where we write $\theta_{ij}$ for the angle between $q_j$ and $x^{(i)}_j-\hat x^{(i)}_j$. 

We've reached a point where we are stuck without further assumption on the distribution $\mathcal D$. 
We do not want to impose strong conditions on it as it is quite outside our control. 
However, not too far of a stretch seems to assume that $\mathcal D$ is *isotropic*, ie, does not 
depend on the direction. Under that assumption, we obtain:

\begin{align*}
\expec_{q\sim\mathcal D}\left[\sum_{i=1}^n\left(\trsp q_jx^{(i)}_j-\trsp q_j\hat x^{(i)}_j\right)^2\right]
&=C\sum_{i=1}^n||x^{(i)}_j-\hat x^{(i)}_j||^2,
\end{align*}

for some constant $C>0$ that only depends on $\mathcal D$ but not on $\hat x^{(i)}_j$.

Recall that each $\hat x^{(i)}_j$ is one of $c_{j1},\ldots,c_{jk}$. Thus 

$$
\argmin_{c_{j1},\ldots,c_{jk}} \expec_{q\sim\mathcal D}\left[\sum_{i=1}^n\left(\trsp q_jx^{(i)}_j-\trsp q_j\hat x^{(i)}_j\right)^2\right]
= \argmin_{c_{j1},\ldots,c_{jk}} \sum_{i=1}^n\min_{s}||x^{(i)}_j-c_{js}||^2,
$$

which is nothing else than the $k$-means objective! 

What does that mean? We split each database vector $x^{(i)}$ into chunks of size $\tfrac{d}{m}$, and then, for each $j=1,\ldots, m$, 
solve for the data $x^{(1)}_j,\ldots, x^{(n)}_j$ a $k$-means clustering problem, resulting 
in the centres $c_{j1},\ldots,c_{jk}$; finally we replace each $x^{(i)}_j$ with the nearest cluster centre $c_{js}$.

Replacing vectors in a dataset by simpler vectors is  called *vector quantisation*[^defvecquant], and also used as a compression tool, 
for example, in video or audio codecs. 
% see wikipedia... vector quantization 

[^defvecquant]: {-} vector quantisation

The approach outlined here can be improved upon. Indeed, I've cheated with the objective {eq}`vqobj`: The objective
incorporates *all* scalar products $\trsp q\hat x^{(i)}$. In the applications, however, we just want to find
the datapoint with largest scalar product with the query, or rather, the top-10 datapoints with largest scalar product. 
Thus, it does not matter if $\trsp q\hat x^{(i)}$ deviates from $\trsp qx^{(i)}$ as long as both scalar products 
are not too large. This insight can be used to devise a more complicated loss function that results in 
better performance; see Guo et al. (2020).[^guo]

[^guo]: *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, 
F. Chern and S. Kumar (2020), [arXiv:1908.10396](https://arxiv.org/pdf/1908.10396)

%Accelerating Large-Scale Inference with Anisotropic Vector Quantization
%Ruiqi Guo*, Philip Sun*, Erik Lindgren*, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar
%arXiv:1908.10396

% https://github.com/google-research/google-research/tree/master/scann

