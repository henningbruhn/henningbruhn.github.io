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

(consproofsec)=
Proof of consistency theorem
----------------------------

%%%% Proof extracted from Bartlett et al, looks quite different there

We prove {prf:ref}`consistencythm`[^BJM03] by establishing a number of claims.
We repeat the theorem here:

```{prf:Theorem} Bartlett, Jordan and MacAuliffe
:label: consistencythmX
:nonumber:
Let $\phi:\mathbb R\to \mathbb R_+$ be convex, continuous and differentiable at 0 with $\phi'(0)<0$.
Then $\phi$ is Bayes-consistent.
```

[^BJM03]: *Convexity, Classification, and Risk Bounds*, P.L. Bartlett, M.I. Jordan and J.D. MacAuliffe (2003)


We define the probability that a given $x\in\mathcal X$ is class 1 as 
$\eta(x)=\proba[y=1|x]$. We claim that: 

```{math}
:label: cons1
L(f)-\bayerr = \expec_x[1_{\sgn f(x)\neq\sgn (2\eta(x)-1)}|2\eta(x)-1|]
```
To prove the claim, 
we first note that 

$$
h_\text{Bayes}:\mathcal X\to\{-1,1\},\, x\mapsto \sgn(2\eta(x)-1)
$$

is a Bayes classifier; see {eq}`bayclass` in [Section Bayes Error](sec:bayeserr). 
To compute the true risk I will pretend that the distribution $\mathcal D$ is discrete; for 
a continuous distribution the computation is very similar.
\begin{align*}
L(f)-\bayerr & = \sum_{(x,y)\in\mathcal X\times\{-1,1\}} \proba[(x,y)] \\
&\qquad\qquad\qquad\qquad \cdot\left(\ell_{0-1}(y,\sgn f(x))-\ell_{0-1}(y,h_\text{Bayes}(x))\right) \\
& = \sum_{x\in\mathcal X} \proba[x] \\
&\qquad\quad \cdot\Big(\eta(x)\left(\ell_{0-1}(1,\sgn f(x))-\ell_{0-1}(1,h_\text{Bayes}(x))\right) \\
&\qquad\quad +(1-\eta(x))\left(\ell_{0-1}(-1,\sgn f(x))-\ell_{0-1}(-1,h_\text{Bayes}(x))\right) \Big)
\end{align*}
Here, we have used that $1-\eta(x)=\proba[y=-1|x]$. Also note that the sum needs to range only
over those $x\in\mathcal X$, for which the Bayes classifier and $\sgn \circ f$ yield different
predictions. That is, we only need to range over $x\in\mathcal X$ with $\sgn f(x)\neq\sgn (2\eta(x)-1)$.

Next, consider a fixed $x\in\mathcal X$ and first consider the case that $\sgn(2\eta(x)-1)=1$. 
Then, $h_\text{Bayes}(x)=1$ and $f$ only yields a different prediction if $\sgn f(x)=-1$. Thus, the 
summand for $x$ above gives
\begin{align*}
&\eta(x)\left(\ell_{0-1}(1,-1) - \ell_{0-1}(1,1)\right)+(1-\eta(x))\left(\ell_{0-1}(-1,-1) - \ell_{0-1}(-1,1)\right) \\
= & \eta(x) -(1-\eta(x)) = 2\eta(x) -1 = |2\eta(x)-1|,
\end{align*} 
as $2\eta(x)-1\geq 0$.

Now consider the case when $\sgn(2\eta(x)-1)=-1$. Then the summand becomes

$$
\eta(x) (0-1)+ (1-\eta(x))(1-0) = 1-2\eta(x) = |2\eta(x)-1|,
$$

as $1-2\eta(x)\geq 0$.

Returning back to $L(f)-\bayerr$, we get
\begin{align*}
L(f)-\bayerr & = \sum_{x\in\mathcal X} \proba[x] 1_{\sgn f(x)\neq\sgn(2\eta(x)-1)} |2\eta(x)-1| \\
&= \expec_x[1_{\sgn f(x)\neq\sgn (2\eta(x)-1)}|2\eta(x)-1|]
\end{align*}
This finishes the proof of {eq}`cons1`.

For the next claim, define for numbers $\alpha\in\mathbb R$ and $\mu\in [0,1]$

$$
C_\mu(\alpha) = \mu\phi(\alpha) + (1-\mu) \phi(-\alpha)
$$

We claim that:
```{math}
:label: cons2
L_\phi(f) - \epsilon_\phi = \expec_x[C_{\eta(x)}(f(x))-\inf_\alpha C_{\eta(x)}(\alpha)]
```
For the proof observe that 
\begin{align*}
L_\phi(f) & = \expec_{(x,y)\sim\mathcal D} \phi(yf(x)) \\
& = \sum_{x\in\mathcal X} \proba[x]\sum_{y\in\{-1,1\}} \proba[y|x] \phi(yf(x)) \\
& = \expec_x \left[\eta(x)\phi(f(x)) +(1-\eta(x))\phi(-f(x))\right] = \expec_x[C_{\eta(x)}(f(x))]
\end{align*}
In a similar way, we get
\begin{align*}
\epsilon_\phi & = \inf_g L_\phi(g) \\
& = \inf_g \expec_x\left[\eta(x)\phi(g(x)) +(1-\eta(x))\phi(-g(x))\right]
\end{align*}
For each $x$ the last expression is minimised if $g(x)$ is chosen such that 
$\eta(x)\phi(g(x)) +(1-\eta(x))\phi(-g(x))$ is as small as possible. 
Thus
\begin{align*}
\epsilon_\phi & = \expec_x\left[\inf_\alpha\eta(x)\phi(\alpha) +(1-\eta(x))\phi(-\alpha)\right] \\
& = \expec_x[\inf_\alpha C_{\eta(x)}(\alpha)]
\end{align*}
This completes the proof of {eq}`cons2`.

Next, we claim that:
```{math}
:label: cons3
\inf_\alpha C_\mu(\alpha) =  \inf_\alpha C_{1-\mu}(\alpha)\quad\text{for all }\mu\in [0,1]
```
This is easy. Note that for every $\mu\in[0,1]$ and $\alpha\in\mathbb R$:
\begin{align*}
C_\mu(\alpha) & = \mu\phi(\alpha) + (1-\mu)\phi(-\alpha) \\
& =(1-\mu)\phi(-\alpha) + (1-(1-\mu))\phi(-(-\alpha))\\& = C_{1-\mu}(-\alpha) 
\end{align*}
Taking the infimum on both sides yields {eq}`cons3`.

We define a somewhat arbitrary looking function $\psi:[0,1] \to\mathbb R$ as

$$
\psi(\theta) = \phi(0) -\inf_\alpha C_{\frac{1+\theta}{2}}(\alpha)
$$

We note that:
```{math}
:label: cons4
\text{$\psi$ is continuous and convex}
```
That $\psi$ is continuous follows from the fact that $(\mu,\alpha) \mapsto C_\mu(\alpha)$
is continuous. To see that $\psi$ is convex, it suffices to see that 

$$
\theta\mapsto \sup_\alpha \left(-C_{\frac{1+\theta}{2}}(\alpha)\right) 
$$

is convex. By {prf:ref}`suplem` this is the case, as for every $\alpha$ the function $\theta\mapsto -C_{\frac{1+\theta}{2}}(\alpha)$
is affine and thus convex.


Next claim:
```{math}
:label: cons5
\psi(\theta)>0\quad\text{whenever }0<\theta\leq 1 
```
For this claim we need  a basic calculus lemma that essentially says that whenever a differentiable 
function has a point with negative derivative then there is a point to the right of the first point with
smaller function value; see {prf:ref}`basiccalclem` in the appendix. 

Consider a fixed $\theta\in (0,1]$ and put $\mu=\tfrac{1+\theta}{2}$. Then $\tfrac{1}{2}<\mu\leq 1$
and $\psi(\theta) = \phi(0)-\inf_\alpha C_\mu(\alpha)$. 
Moreover, 
\begin{align*}
C'_{\mu}(\alpha) & = \frac{\partial}{\partial\alpha}\left( \mu\phi(\alpha)+(1-\mu)\phi(-\alpha)\right) \\
& =  \mu\phi'(\alpha) - (1-\mu) \phi'(-\alpha) 
\end{align*}
and thus $C'_{\mu}(0) = (2\mu-1) \phi'(0) <0 $ as $2\mu-1>0$ and $\phi'(0)<0$ by assumption. 

Now we use the basic calculus {prf:ref}`basiccalclem` to deduce that there is an $\alpha_0>0$ with

$$
C_\mu(\alpha_0)\leq C_\mu(0)+\tfrac{\alpha_0}{2}C'_\mu(0) < C_\mu(0)
$$

On the other hand, 

$$
C_\mu(0) = \mu\phi(0) + (1-\mu) \phi(-0) = \phi(0)
$$

Thus, we get $C_\mu(\alpha_0) < \phi(0)$ and therefore $\inf_\alpha C_\mu(\alpha)<\phi(0)$,
which means that 

$$
\psi(0) = \phi(0)- \inf_\alpha C_\mu(\alpha) > \phi(0)-\phi(0) = 0
$$

This proves {eq}`cons5`.

We come to the first major result in the proof:
```{math}
:label: cons6
\text{for any }\theta_1,\theta_2,\ldots \in[0,1]\text{ with }\psi(\theta_i)\to 0
\text{ it follows that }\theta_i\to 0
```
Suppose that $\theta_i\not\to 0$. Then by going to a subsequence we may assume
that $\theta_i\to c>0$. As $\psi$ is continuous 

$$
\lim_{i\to\infty} \psi(\theta_i) = \psi\left(\lim_{i\to\infty}\theta_i\right) =\psi(c) >0,
$$

where the last inequality is {eq}`cons5`. This, however, is a contradiction to $\psi(\theta_i)\to 0$.

Next we claim for every $\alpha\in\mathbb R$ and $\mu\in [0,1]$ that 
```{math}
:label: cons7
C_\mu(\alpha)\geq \phi(0) \text{ if }\sgn(\alpha)\neq \sgn(2\mu-1)
```
As $\phi$ is convex and differentiable at 0 it follows from {prf:ref}`gradlem` that 
\begin{align*}
\phi(0)+\alpha \phi'(0) & \leq \phi(\alpha) \\
\phi(0)-\alpha \phi'(0) & \leq \phi(-\alpha) 
\end{align*}
This yields
\begin{align*}
C_\mu(\alpha) & = \mu\phi(\alpha) + (1-\mu) \phi(-\alpha) \\
& \geq \phi(0) + \alpha (2\mu-1) \phi'(0) \geq \phi(0),
\end{align*}
as $\alpha (2\mu-1)\leq 0$ by assumption, and $\phi'(0)<0$.
This finishes the proof of {eq}`cons7`.

The final claim before we can finish the proof: 
```{math}
:label: cons8
\psi\left(L(f)-\bayerr\right)\leq L_\phi(f)-\epsilon_\phi\quad\text{for all }f:\mathcal X\to\mathbb R
```
We use {eq}`cons1` and convexity {eq}`cons4` of $\psi$ and [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) to see that 
\begin{align*}
\psi(L(f) -\bayerr) & = \psi\left(\expec_x[1_{\sgn f(x)\neq\sgn (2\eta(x)-1)}|2\eta(x)-1|]\right) \\
& \leq \expec_x[1_{\sgn f(x)\neq\sgn (2\eta(x)-1)}\psi(|2\eta(x)-1|)]
\end{align*}

What can we say about $\psi(|2\eta(x)-1|)$? 
We use that $\inf_\alpha C_\eta(\alpha)$ is symmetric around $\tfrac{1}{2}$, see {eq}`cons3`, to write

$$
\psi(|2\eta(x)-1|) = \phi(0)- \inf_\alpha C_{\eta(x)}(\alpha)
$$

Moreover, for every $x$ with $\sgn f(x)\neq\sgn(2\eta(x)-1)$ we know, by {eq}`cons7`, that 
$C_{\eta(x)}(f(x))\geq \phi(0)$ and thus
\begin{align*}
\psi(L(f) -\bayerr)& \leq \expec_x[1_{\sgn f(x)\neq\sgn (2\eta(x)-1)} (\phi(0)- \inf_\alpha C_{\eta(x)}(\alpha)) ]\\
& \leq \expec_x[1_{\sgn f(x)\neq\sgn (2\eta(x)-1)} (C_{\eta(x)}(f(x)) - \inf_\alpha C_{\eta(x)}(\alpha)) ] \\
& \leq \expec_x[ C_{\eta(x)}(f(x)) - \inf_\alpha C_{\eta(x)}(\alpha) ] \\
& = L_\phi(f) -\epsilon_\phi,
\end{align*}
where the last equality is due to {eq}`cons2`.

We can finally complete the proof of {prf:ref}`consistencythm`. 
Let a sequence $f_1,f_2,\ldots $ of functions $f_i:\mathcal X\to\mathbb R$ with 

$$
L_\phi(f_i)\to \epsilon_\phi\text{ as }i\to\infty
$$

be given. Define $\theta_i=L(f_i)-\bayerr$, and note that $\theta_i\in[0,1]$ as the zero-one loss 
is either $1$ or $0$.
Then, by {eq}`cons8`, $\psi(\theta_i)\leq L_\phi(f_i)-\epsilon_\phi\to 0$. 
By {eq}`cons6` this implies $\theta_i\to 0$, ie, $L(f_i)\to\bayerr$. 

