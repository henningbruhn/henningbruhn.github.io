---
orphan: true
---

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

(sec:corr)=
Example of correlated classifiers
---------------------------------


%%%%% this would also make for a good exercise %%%%%
%%%%% if only [1,2]^3 is class 1 then, no wisdom of the crowd, and pw correlation coeff -1/15 %%%%%
%%%%% other classifiers: h_i: 1 if x_i\geq 1 -> correlation coeff -1/3 and wisdom of the crowd

Let's look at an example. We fix the domain to $\mathcal X=[0,2]^3$,
and the distribution $\mathcal D$ so that the marginal probability $\proba[x]$
is uniform on $\mathcal X$, and so that the data completely determines the class.
We set

$$
\proba[1|x]=\begin{cases}
1 & \text{ if }x\in[1,2]^3\\
1 & \text{ if }x\in[0,1]\times[1,2]\times[1,2]\\
1 & \text{ if }x\in[1,2]\times[0,1]\times[1,2]\\
1 & \text{ if }x\in[1,2]\times[1,2]\times[0,1]\\
0 & \text{ otherwise}
\end{cases}
$$

and $\proba[-1|x]=1-\proba[1|x]$ for all $x\in[0,2]^3$.
In the [figure](ensembleexfig) on the left, the area with class 1 is marked in grey.
The two right panels in the figure show the domain in two 2-dimensional slices
perpendicular to the $x_3$-axis. Again, grey marks the area of class 1. 

```{figure} pix/correlated.png
:name: ensembleexfig
:height: 6cm

Class 1 in grey. Three digit binary strings encode whether $h_1,h_2,h_3$ are correct
on the corresponding cube.
```

We consider three classifiers $h_1,h_2,h_3:\mathcal X\to\{-1,1\}$ defined as

$$
h_i(x)=\begin{cases}
1& \text{ if }x_j\geq 1\text{ for some }j\neq i\\
-1 & \text{ otherwise}
\end{cases}
$$

That is, $h_1$ for example, predicts -1 on $[0,2]\times[0,1]^2$, and $+1$
everywhere else. 
The associated binary random variables $X_1,X_2,X_3$ record whether 
the classifiers are correct: $X_i(x)=1$ if and only if $h_i$ correctly 
predicts the class of $x$. 
On the right in [the figure](ensembleexfig) we see the behaviour of the $X_i$ 
expressed in three digit binary numbers. For instance, in the upper right corner
of the rightmost panel, all three classifiers are correct, resulting in $111$.
In contrast, in the lower left corner of the panel, ie, in $[0,1]^2\times[1,2]$, 
the  classifiers $h_1$ and $h_2$ predict +1, which is false, while $h_3$ predicts
the correct class -1, resulting in 001.

Let's start calculating. First, the domain $\mathcal X$ splits naturally into the eight cubes
$[i,i+1]\times[j,j+1]\times[k,k+1]$ with $i,j,k\in\{0,1\}$. For each of the eight cubes the 
probability of drawing an $x$ from it is exactly $\tfrac{1}{8}$.

Checking the [figure](ensembleexfig), we see that the classifier $h_1$ is correct
on six of the eight cubes. Thus

$$
\expec[X_1=1]=\proba[X_1=1]=\tfrac{6}{8} = \tfrac{3}{4}
$$
Due to symmetries, the expectation is the same for $X_2,X_3$.

With the [figure](ensembleexfig) it is also easy to determine how well the majority classifier
performs. In fact, we only need to count the cubes with at least two $1$s. 

$$
\proba[\text{majority correct}]=\tfrac{5}{8}
$$

We observe that the majority classifier performs worse than each individual classifier!
Why is that? Because of stocastic dependence.
 Let's check.

Again, with the [figure](ensembleexfig) we see that there are precisely five of the eight cubes, where 
both $h_1$ and $h_2$ are correct. Those are the ones labelled $11b$, for $b\in\{0,1\}$. 
Thus $\proba[X_1=1\text{ and }X_2=1]=\tfrac{5}{8}$. On the other hand

$$
\proba[X_1=1]\cdot \proba[X_2=1]= \tfrac{3}{4}\cdot\tfrac{3}{4} < \tfrac{5}{8}.
$$

What about the correlation coefficient? We calculate

$$
\expec[X_1X_2] = \proba[X_1=1\text{ and }X_2=1]=\tfrac{5}{8}
$$

and

$$
\vari[X_1]=\vari[X_2]=\expec[X_2^2]-\expec[X_2]^2= \expec[X_2]-\expec[X_2]^2 
=\tfrac{3}{4}-\tfrac{9}{16}= \tfrac{3}{16},
$$

and thus

$$
\text{corr}(X_1,X_2)=\frac{\expec[X_1X_2]-\expec[X_1]\expec[X_2]}{\sqrt{\vari[X_1]\vari[X_2]}}
= \frac{\frac{5}{8}-\left(\frac{3}{4}\right)^2}{\frac{3}{16}} = \frac{1}{3}
$$
% 10 -9 

We see that the positive pairwise correlation is responsible for the meagre accuracy of the
majority classifier. 



