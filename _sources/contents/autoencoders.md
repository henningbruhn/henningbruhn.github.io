$\newcommand{\trsp}[1]{#1^\intercal} % transpose
\newcommand{\id}{\textrm{Id}} % identity matrix
\newcommand{\rank}{\textrm{rank}}
\DeclareMathOperator*{\expec}{\mathbb{E}} % Expectation
\DeclareMathOperator*{\proba}{\mathbb{P}}   % Probability
\newcommand{\KL}{\textrm{D}_\textrm{KL}} % Kullback-Leibler divergence
\DeclareMathOperator*{\argmax}{argmax}
$

Autoencoders
============

Labelled training data is often scarce. Labelling is typically costly and time consuming. Often, however, 
unlabelled data is much more readily available (think all the images on the internet, or all of wikipedia). 
*Autoencoders* are a way to leverage unlabelled data 
in tasks that in principle would require labelled data. So what's an autoencoder? An autoencoder
consists of two parts: an *encoder* neural network $e:\mathbb R^n\to\mathbb R^k$ and 
a *decoder* neural network $d:\mathbb R^k\to\mathbb R^n$. Here, $n$ is the input dimension, while 
$k$ is the dimension of the *latent space*, ie, of the *latent representation* $z=e(x)$. 
The idea of an autoencoder is that it learns to replicate the input. That is, that on input $x$ it computes

$$
d(e(x))=\hat x \approx x
$$

For this reason, an autoencoder is often trained with MSE loss $||x-\hat x||_2$.

```{figure} pix/autoencoder.png
:name: autoencoderfig
:height: 6cm

An autoencoder.
```


Clearly, nothing is gained if the autoencoder simply learns the identity function. For this reason, the latent dimension 
is often much smaller than the input dimension, ie, $k\ll n$. The autoencoder should also not overfit, should 
not be able to memorise the whole training set. Ideally, the autoencoder learns a meaningful, low-dimensional 
latent representation of the data. 

What's an autoencoder good for?

* *semi-supervised* learning: This is the setting alluded to above. In many classification tasks 
labelled data may not be abundant. For example, we may want to classify user feedback as generally positive
or negative. To obtain the labels, positive or negative, we may need to pay human workers to manually read and 
classify many user comments. Unlabelled user feedback, however, may be freely available in large quantities.
In this situation, an autoencoder can be trained to learn useful features about user feedback. Then, 
we take the encoder and a new, small neural network $f:\mathbb R^k\to [0,1]$ that we put on top of the
encoder. ($f$ maps to $[0,1]$ as I am assuming a simple binary classification task.) The combined 
network $f\circ  e$ is then trained on the labelled data. If necessary, the weights of $e$ could even 
be fixed during that second round of training. 

    In this way, it is possible to train a relatively powerful neural network (with many weights) even if
    only a small number of labelled data is available.

* *anomaly detection*: It has become fairly cheap to record large amounts of machine data during the 
manufacturing of goods. Companies have an interest in detecting early whether manufactured goods
are in perfect order and can be sold, or whether the production process resulted in faults, so that the
goods need to be repaired or discarded. The hope is to detect that via the recorded machine data. This may
be voltage time series data, sound data or data on forces that some part of the machine was subjected to. 
Anomaly detection then aims to detect data, ie, goods, that somehow look different than the typical data.
(Normally, the vast majority of the production will be fault-free.) Autoencoders can do that. The idea is
that an autoencoder learns to reproduce the typical data to a high degree, so that the *reconstruction error*
$||\hat x - x||$ is small. Anomalous data, however, will be difficult to replicate, so that 
the resulting reconstruction error is large. 

* *denoising*: Image or sound data may in some applications be regularly corrupted by noise. 
Autoencoders can be used to remove that noise. For this, each training data point $x$ is subjected 
to some noise $\epsilon$ before being fed into the 
autoencoder, so that the autoencoder learns to minimise $||e(d(x+\epsilon))-x||$. 
Once trained, the autoencoder will, on input of noisy data output denoised data.

In general, the encoder of an autoencoder may be seen as a *dimension reduction* method, and 
thus can be compared to classic methods such as *principle component analysis*, or *PCA* for 
short. In fact, the connection to PCA is quite a bit tighter.


````{dropdown} Cost of labelling
:color: success
:icon: telescope

Obtaining reliable labels for data may be time-consuming and costly,
especially, if the data needs to be labelled by experts or if the data are proprietary and cannot be made
publicly available. If that is not the case, then the data may be labelled through one 
of many micro-work services such as Amazon Mechanical Turk. MTurk, probably the most well-known 
micro-work platform, apparently charges shockingly small fees, on the order of \$0.012 per image label. 
There is, however, increasing concern whether labellers are treated fairly. 

[MTurk pricing](https://aws.amazon.com/sagemaker/groundtruth/pricing/) accessed in 2024\
[http://faircrowd.work/](http://faircrowd.work/)
% Apparently big German crowdsourcing service: Clickworker
````

Principle component analysis
----------------------------

Principle component analysis is a common method to project some data 
from a high-dimensional space $\mathbb R^n$ to a low-dimensional space $\mathbb R^k$,
while keeping as much information on the data as possible. 
Let's briefly
recap what PCA does and how it works. 

PCA is based on *singular value decomposition*, or *SVD*. 
For this consider a matrix $X\in\mathbb R^{N\times n}$, which in our context will 
arise from stacking the data $x^{(1)},\ldots,x^{(N)}\in\mathbb R^n$  as row vectors on top of each other, ie,
each row corresponds to a data point. (The definition of the SVD works for any matrix, though.)
It's best to assume $N\gg n$.

The SVD of $X\in\mathbb R^{N\times n}$ is then the decomposition

$$
X=U\Sigma \trsp V,
$$

where $U\in\mathbb R^{N\times N}$, $V\in\mathbb R^{n\times n}$ are orthonormal matrices, ie, matrices with $U\trsp U=\id$ and $V\trsp V=\id$, 
while $\Sigma\in\mathbb R^{N\times n}$ has the following form

$$
\Sigma=\begin{cases}
\begin{pmatrix}\text{diag}(\sigma)\\ 0 \end{pmatrix} & \text{ if }n\leq N\\
\begin{pmatrix}\text{diag}(\sigma)& 0 \end{pmatrix} & \text{ if }n\geq N
\end{cases}
$$

where $\text{diag}(\sigma)$ denotes a diagonal matrix with entries $\sigma=(\sigma_1,\ldots,\sigma_p)$,
$p=\min(n,N)$. Finally, the *singular values* $\sigma_i$ are non-negative and ordered by size: 
$\sigma_1\geq \sigma_2\geq\ldots\geq\sigma_p\geq 0$. 


What is now the PCA? For the *PCA*, we fix an integer $k\leq n$, the dimension of the 
projection. Then, we define
$U_k\in\mathbb R^{N\times k}$ and $V_k\in\mathbb R^{n\times k}$ to consist of the first $k$ columns of 
$U$ and $V$, and we define $S_k\in\mathbb R^{k\times k}$ as $\text{diag}(\sigma_1,\ldots,\sigma_k)$.
The PCA (with $k$ principle components) is then the projection $\mathbb R^n\to\mathbb R^k$ defined by 
$x\mapsto \trsp{V_k} x$ (here, $x$ is seen as a column vector). Thus, the latent representation of $X$ becomes
$Z=XV_k$ (note the lack of transposition as $X$ contains the data as row vectors).
Note that 
\begin{align*}
Z& =XV_k=U\Sigma \trsp VV_k = U\begin{pmatrix}\text{diag}(\sigma) \\0\end{pmatrix} \begin{pmatrix}\id_k \\0\end{pmatrix} \\
& = U \begin{pmatrix}\text{diag}(\sigma_1\ldots \sigma_k) \\0\end{pmatrix} = U_kS_k
\end{align*}

It is also possible to recover data in the original space $\mathbb R^n$ from the latent representation in $\mathbb R^k$,
as follows: $\mathbb R^k\to\mathbb R^n$, $z\mapsto V_kz$.
For the data matrix, this translates to $\hat X=Z\trsp{V_k}$ and thus to

$$
\hat X =Z\trsp{V_k} = U_kS_k\trsp{V_k}
$$

Note that $\rank(\hat X)\leq k$ as $S_k\in\mathbb R^{k\times k}$. 

In fact it turns out that $\hat X =Z\trsp{V_k} = U_kS_k\trsp{V_k}$ is the best approximation of $X$ among 
all matrices of rank at most $k$. This is the [Eckhart-Young-Mirsky](https://en.wikipedia.org/wiki/Low-rank_approximation) theorem:

```{prf:Theorem} Eckhart-Young-Mirsky
:label: eymthm
Let $X\in\mathbb R^{n\times N}$ be a matrix, and let $k\leq\min(n,N)$ be an integer. 
Let $\hat X= U_k\text{diag}(\sigma_1\ldots\sigma_k)\trsp{V_k}$, where $X=U\Sigma \trsp V$ is the SVD of $X$ and $\sigma$ 
is the vector of singular values.
Then 

$$
||\hat X -X||_F \leq ||A-X||_F,
$$ 
for every matrix $A\in\mathbb R^{n\times N}$ with $\rank(A)\leq k$.
```
(Recall that $||A||_F$ is the [Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html), ie, the $2$-norm of $A$ written as a vector.)


PCA and autoencoder
-------------------

% see also
% Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima 
% PIERRE BALDI AND KURT HORNIK, Neural Networks, Vol. 2, pp. 53-58, 1989
We consider the simplest autoencoder possible: an encoder that consists of a neural network *without* hidden layers, and with 
linear activation function, and the same holds for the decoder. On top of that, we set the bias to 0 everywhere.
Then, the encoder does nothing more than a matrix-vector mulitplication: $e:x\mapsto Ex$ for some matrix $E\in\mathbb R^{k\times n}$.
The same goes for the decoder: $d:z\mapsto Dz$ for some $D\in\mathbb R^{n\times k}$. 
In total, the autoencoder computes $x\mapsto DEx$. 

If we collect the (training) data as above in a matrix $X\in\mathbb R^{N\times n}$ (with each data point as a row)
then the autoencoder computes $\hat X = X\trsp E\trsp D$. Note that the matrix $C=\trsp E\trsp D$ has rank at most $k$,
as each of $D,E$ has one dimension equal to $k$. 

The autoencoder is trained with SGD. That is, SGD aims to find weights $D,E$ such that

$$
\text{loss} = \sum_{i=1}^N||x^{(i)}-DEx^{(i)}||_2^2 = ||X-\hat X||_F^2
$$

is as small as possible. While we are not guaranteed that SGD finds the global minimum we see
with {prf:ref}`eymthm` that the minimum is achieved by the SVD (as $\rank(C)\leq k$, whatever the weights). 
Thus, in the best case the autoencoder will basically compute the PCA. 
(This is not quite right. We're not guaranteed that the autoencoder finds the PCA projection. 
Technically, we're only guaranteed that the reconstruction error of PCA and of the autoencoder 
coincide -- if the autoencoder is optimally trained.)

[^autocode]
Nobody will use such a simple autoencoder. So, why is that insight interesting? 
We can interpret that as: In the simplest setting the autoencoder is as good as 
PCA, which is a venerable and widely used technique; with a more powerful encoder and decoder 
(ie, with hidden layers and non-linear activation), autoencoders are likely 
to be better still. 

[^autocode]: {-} [{{ codeicon }}auto-vs-pca](https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/autoencoder/auto-vs-pca.ipynb)

(VAEsec)=
Variational autoencoders
------------------------

The latent representation in autoencoders makes it possible to generate new samples. 
Generative AI such as midjourney, DALL-E and stable diffusion
show that this may be incredibly useful. (These systems, however, are based on a different method.)
 There is just a problem -- how would we sample
from the latent space? We do not have any control over the latent space, and we have no idea
how the latent representations of the samples are distributed. 

[^king19]
*Variational autoencoders* (VAEs) aim to impose tight control on the latent space. 
In fact in a variational autoencoder, the latent vectors are forced to conform
to a fixed probability distribution, normally to a multivariate normal distribution. 
Then, it's easy to sample from the latent space: We simply draw $z\sim \mathcal N(0,1)$,
and then use the decoder to compute a new sample $d(z)$. 

[^king19]: {-} *An Introduction to Variational Autoencoders*, D.P. Kingma and M. Welling (2019),
[arXiv:1906.02691](https://arxiv.org/abs/1906.02691)

In general a variational autoencoder is an autoencoder, consisting of an encoder $e$,
a decoder $d$, and some stochastic elements. First, though, let's take a step 
back and let's look at the motivation for them. 

We assume that there is some unknown process that generates samples $x$.
To make this a bit more concrete, let us concentrate on cat pictures. 
Thus, we assume that cat pictures $x$ are generated by a distribution $x\sim p^*(x)$.
Here, $p^*$ is a probability density function (that is unknown and likely very complicated). 
The stochastic process is actually governed by hidden, latent variables $z$. That is, 
we imagine that the process to generate a sample $x$ has two steps:

* first, a latent vector $z\sim p^*(z)$ is drawn, where we require that
the distribution on the latent space is equal to the standard normal distribution,
ie 

    $$
    p^*(z)=\frac{1}{(2\pi)^\frac{k}{2}}e^\frac{||z||_2^2}{2}
    $$

* then, the sample $x$ is drawn, conditioned on $z$: $x\sim p^*(x|z)$.

 
This has only shifted the problem, as we 
we do not know $p^*(x|z)$. We therefore will try to approximate 
$p^*(x|z)$ with a parametric probability density function $p_\theta(x|z)$
that still will be quite complicated. *Parametric* here means that 
the function depends on the parameter vector $\theta$. Essentially, 
$p_\theta(x|z)$ will be determined by the decoder of the VAE, and $\theta$
will simply collect all the weights of the neural network that realises
the decoder. This is coupled with a simple probabilistic distribution;
this may be a multivariate Bernoulli distribution or a multivariate 
normal distribution. Here we fix a normal distribution.

```{figure} pix/VAE1.png
:name: vaeXfig
:width: 15cm

From latent space to sample.
```


More concretely, 

1. $z\sim\mathcal N(0,1)$ is drawn from the $k$-dimensional standard normal distribution; 

2. the decoder network computes two vectors $\mu^{(1)},s^{(1)}$, ie $d(z)=(\mu^{(1)},s^{(1)})$;

3. $x$ is drawn from $\mathcal N(\mu^{(1)},s^{(1)})$.

A bit of fineprint: Here, we assume that the final output $x$ is drawn from a normal distribution $\mathcal N(\mu^{(1)},s^{(1)})$.
For different types of data, a different final distribution may be appropriate. For instance, for image data, 
we might have a multivariate Bernoulli distribution instead of the normal distribution. We stick 
with  $\mathcal N(\mu^{(1)},s^{(1)})$ to keep things simple.

The steps 1.-3. describe how we generate new samples. What is not yet clear, is how we can train 
the decoder in the first place. For that, we also need to have a closer look at the encoder. 
For that, 
we approximate the probability density function $p^*(z|x)$ by 
a parametric PDF $q_\phi(z|x)$. This time, we use the encoder of the autoencoder, and 
the parameters $\phi$ will be the weights of the encoder. 
Indeed, for the encoder we prescribe the following steps: 

1. on input $x$, the encoder computes two vectors $\mu^{(2)},s^{(2)}$, ie $e(x)=(\mu^{(2)},s^{(2)})$;

2. then $\epsilon\in\mathbb R^k$ is drawn $\epsilon\sim\mathcal N(0,1)$; and 

3. finally, we compute a latent representation $z=\mu^{(2)}+(s^{(2)}\odot \epsilon)$, 
where $\odot$ denotes entry-wise multiplation.

Why don't we draw $z\sim\mathcal N(\mu^{(2)},s^{(2)})$, in a similar way as for the decoder? 
Well, we effectively do that, just in a slightly roundabout way. Below we will see that there is a good 
reason to pursue this three step process.


```{figure} pix/VAE2.png
:name: vaefig
:width: 15cm

A variational autoencoder.
```

How do we determine the weights of the autoencoder? Let us start with the decoder. 
We assume we have access to training data $x^{(1)},\ldots, x^{(N)}$ that is drawn
iid according to the hidden process $p^*$. We do a maximum likelihood fit:

```{math}
:label: thetastar
\begin{align}
\theta^* & = \argmax_{\theta} \prod_{i=1}^Np_\theta(x^{(i)}) \notag \\
\Leftrightarrow\quad\theta^* & = \argmax_\theta \sum_{i=1}^N\log\left(p_\theta(x^{(i)})\right) 
\end{align}
```
Can we maximise the expression on the right-hand side? Not directly. In fact, we cannot even 
evaluate $p_\theta(x)$ directly, for any $x=x^{(i)}$.
Rather, what we can do is to compute $p_\theta(x|z)$ for any given $z$: As outlined
above the decoder network yields parameters that are then fed into a simple stochastic process. 
So, let's rewrite $p_\theta(x)$ as

$$
p_\theta(x)=\int_z p_\theta(x|z) p_\theta(z)dz
$$

This looks partially better because at least we can compute $p_\theta(x|z)$, and I suppose that 
for $p_\theta(z)$ we could use that we require the latent vectors $z$ to conform to a standard normal distribution. 
However, this is also quite a bit worse, because we would need to evaluate an integral over a complex space. 
It turns out that that is quite infeasible. So, let's take a different route.

Fix $x=x^{(i)}$ for some $i$ and let's continue with $\log p_\theta(x)$. Because 
notation is already fairly dense, I will suppress the indices $\theta$ and $\phi$ for the moment, and 
simply write $p(x)$ and $q(z|x)$ and so on. Then
\begin{align*}
\log p(x) & = \expec_{z\sim q(z|x)} [\log p(x)]
\end{align*}
as $p(x)$ does not depend on $z$, and thus is a constant as far as $z\sim q(z|x)$ is concerned. 
Again, to keep notation uncluttered I will simply write subscript $q$ instead of $z\sim q(z|x)$.
As 

$$
p(z|x)=\frac{p(x,z)}{p(x)} \quad \Leftrightarrow \quad p(x) = \frac{p(x,z)}{p(z|x)}
$$

we obtain
\begin{align*}
\log p(x) & = \expec_{q} [\log p(x,z)-\log p(z|x)] \\
& = \expec_q\left[\log p(x,z)-\log q(z|x) +\log q(z|x) - \log p(z|x) \right] \\
& = \underbrace{\expec_q\left[\log p(x,z)-\log q(z|x)\right]}_{\text{\rm ELBO}_{\theta,\phi}(x)} + \underbrace{\expec_q\left[\log q(z|x) - \log p(z|x) \right]}_{\KL(q(z|x)||p(z|x))}
\end{align*}
The part 

$$
\expec_q\left[\log p(x,z)-\log q(z|x)\right]
$$

is called the *Evidence Lower BOund*, or *ELBO*. I am writing $\text{ELBO}_{\theta,\phi}(x)$ to stress
that the ELBO depends on parameters $\theta$, $\phi$ and also on $x$.
The other part is the [*Kullback-Leibler divergence*](#KLsec).

By Gibb's inequality ({prf:ref}`gibbsineq`), the Kullback-Leibler divergence is always non-negative, which results in 
\begin{align*}
\log p_\theta(x) &  \geq \text{ELBO}_{\theta,\phi}(x)
\end{align*}
Thus, if $\text{ELBO}_{\theta,\phi}(x)$ becomes larger, then so does $\log p_\theta(x)$. 

Originally, our aim was to maximise {eq}`thetastar`, which we argued to be infeasible, at least directly. 
Instead, we solve

```{math}
:label: elbomax
\max_{\theta,\phi} \sum_{i=1}^M\text{ELBO}_{\theta,\phi}\big(x^{(i)}\big)
```
and then take the maximisers $\theta^*$ and $\phi^*$ as weights for the decoder and the encoder of the autoencoder.  

Before we can maximise the ELBO we will be massaging it a bit further. Why? In its current form, 
it contains $p_\theta(x,z)$, and we do not know how to compute that. Thus, we use
$p(x,z)=p(x|z)p(z)$ to replace that term. We get
\begin{align*}
\text{ELBO}_{\theta,\phi}(x) & = \expec_q\left[\log p(x,z)-\log q(z|x)\right] \\
& = \expec_q\left[\log p(x|z) +\log p(z) -\log q(z|x)\right] \\
& = \expec_q\left[\log p(x|z)\right] - \expec_q\left[\log q(z|x) -\log p(z)\right] \\
& = \expec_q\left[\log p(x|z)\right] - \KL(q(z|x)||p(z))
\end{align*}
We make one more simplification. The main motivation for all this was to impose a simple probability distribution,
namely the standard normal distribution $\mathcal N(0,1)$, on the latent space of the $z$. 
As a consequence, $p_\theta(z)$ should, after fitting the parameters, coincide with the standard normal distribution. 
We now assume that is already the case and replace $p_\theta(z)$ directly with $\mathcal N(0,1)$.

Taking the negative, our aim becomes:

```{math}
:label: elbomax2

\min_{\theta,\phi}\Big(-&\text{ELBO}_{\theta,\phi}(x)\Big)\notag \\  
& = \min_{\theta,\phi} \KL(q_\phi(z|x)||\mathcal N(0,1)) -\expec_{z\sim q_\phi(z|x)}\left[\log p_\theta(x|z)\right] 
```

This then is the *loss function* when training the autoencoder. It consists of two parts:

* the *reconstruction loss*: $-\expec_{z\sim q_\phi(z|x)}\left[\log p_\theta(x|z)\right]$. 
This is a bit hard to parse. If $x$ is fed into the VAE then the loss is small if the likelihood 
of $x$ with respect to $p_\theta(x|z)$ is large. We will consider the reconstruction loss further down below.

* the *Kullback-Leibler loss*: $\KL(q_\phi(z|x)||\mathcal N(0,1))$. This is small if $q_\phi(z|x)$ is 
close to the standard normal distribution, and this is precisely what we want. The aim is to get a
distribution on the latent space that is close to the normal distribution.

How then can we minimise the loss? As always with stochastic gradient descent.
We will see below that we can obtain an explicit expression for the KL-loss, so that this loss behaves 
in precisely the same way as other losses with which we train neural networks. 
What about the reconstruction loss? We will need to work a bit more. 


Reconstruction loss
-------------------

To employ stochastic gradient descent on the reconstruction loss we need to compute 
gradients $\nabla_\theta$ and $\nabla_\phi$ with respect to the parameters $\theta,\phi$. 
The gradient with respect to $\theta$ is relatively uncomplicated. In particular, the expectation and $\nabla_\theta$ 
commute as the stochastic process $z\sim q_\phi(z|x) $ is independent of $\theta$:
\begin{align*}
\nabla_\theta \expec_{z\sim q_\phi(z|x)}\left[\log p_\theta(x|z)\right] 
& = \expec_{z\sim q_\phi(z|x)}\left[\nabla_\theta p_\theta(x|z)\right] 
\end{align*}
In particular, $\nabla_\theta p_\theta(x|z)$ is mostly determined by the decoder network (the final probability distribution
also contributes), so that $\nabla_\theta p_\theta(x|z)$ simply becomes the gradient with respect to the weights
of the decoder network.

What about the gradient with respect to $\phi$? It is not clear whether that commutes with the expectation,
as $z\sim q_\phi(z|x)$ does depend on $\phi$. To get around that we use the so-called
*reparametrisation trick*. We recall that drawing $z$ really means drawing $\epsilon\sim\mathcal N(0,1)$
and then computing $z=\mu^{(2)}+s^{(2)}\odot\epsilon$. Thus
\begin{align*}
\nabla_\phi \expec_{z\sim q_\phi(z|x)}\left[\log p_\theta(x|z)\right] 
& =  \nabla_\phi \expec_{\epsilon\sim \mathcal N(0,1)}\left[\log p_\theta(x|\mu^{(2)}+s^{(2)}\odot\epsilon)\right] \\
& = \expec_{\epsilon\sim \mathcal N(0,1)}\left[\nabla_\phi p_\theta(x|\mu^{(2)}+s^{(2)}\odot\epsilon)\right] 
\end{align*}
Recall that $\mu^{(2)}$ and $s^{(2)}$ are computed by the encoder, which depends on the weights $\phi$.
That means in particular, that we can compute the gradient $\nabla_\phi$ in the normal way. 

There is one final issue, namely the expectation $\expec_\epsilon$ in front of the gradient. What should we do with that?
We do a Monte Carlo estimation, which is fancy speak for: draw a number of samples $\epsilon^{(1)},\ldots,\epsilon^{(L)}\sim\mathcal N(0,1)$
and take the mean of the resulting gradient, ie

$$
\nabla \expec_{\epsilon}[\log p_\theta(x|z)]\approx \frac{1}{L}\sum_{i=1}^L\nabla\log p_\theta(x|\mu^{(2)}+s^{(2)}\odot\epsilon^{(i)})
$$

In practice, it seems it is not uncommon to draw a single sample (but each time anew for each training data point), that is, 
to take $L=1$.


To make this a bit more concrete, let us assume that in the final step $x$ is actually drawn 
from a normal distribution $\mathcal N(\mu^{(1)},s^{(1)})$ as described above.  
To make the calculation easier, assume moreover, that $s^{(1)}=\sigma^2\1$ for a scalar $\sigma\in\mathbb R$. 
Then 
the probability density function for the normal distribution $\mathcal N(\mu^{(1)},s^{(1)})$
becomes

$$
\frac{1}{(2\pi\sigma)^{n/2}} e^{-\frac{||x-\mu^{(1)}||^2}{2\sigma^2}}
$$

and thus
\begin{align*}
-\log p_\theta(x|z) & = -\log\left( \frac{1}{(2\pi\sigma)^{n/2}} e^{-\frac{||x-\mu^{(1)}||^2}{2\sigma^2}}\right) \\
& = \frac{n}{2}\log(2\pi\sigma) + \frac{1}{2\sigma^2}||x-\mu^{(1)}||^2 
\end{align*}
This is interesting because we suddenly find the mean square loss $||x-\mu^{(1)}||^2$ loss prominently in there,
exactly as for an ordinary autoencoder. 

The Kullback-Leibler loss
-------------------------

What about the Kullback-Leibler loss $\KL(q_\phi(z|x)||\mathcal N(0,1))$?
We had fixed the distribution $q_\phi(z|x)$ to be a normal distribution. Indeed, it is simply $\mathcal N(\mu^{(2)},s^{(2)})$,
where $\mu^{(2)}$ and $s^{(2)}$ are computed by the encoder network. Fortunately, for two 
normal distributions there is an explizit formula for the Kullback-Leibler divergence
(and in our case, one is even the standard normal distribution). 

```{prf:Theorem}
:label: KLnormalthm

$$
\KL(\mathcal N(\mu,s)||\mathcal N(0,1)) = \tfrac{1}{2}\sum_{i=1}^k\left(s_i+\mu_i^2-1-\ln s_i\right)
$$

for any $\mu\in\mathbb R^k$ and $s\in\mathbb R_{>0}^k$.
```

Before we start with the proof of the theorem let us see what that means for the Kullback-Leibler loss 
of a variational autoencoder. With the theorem, we compute the loss as:

$$
\KL(q_\phi(z|x)||\mathcal N(0,1)) = \tfrac{1}{2}\left(||\mu^{(2)}||_2^2+\sum_{i=1}^ks_i^{(2)}-k-\sum_{i=1}^k2\ln(s^{(2)}_i)\right)
$$

The Kullback-Leibler loss thus acts as a regulariser on the latent space. That is, it institutes a penalty
for large weights. 


To prove the theorem we use an easy insight on the Kullback-Leibler divergence for multivariate distributions.

```{prf:Lemma}
:label: multiKLlem
Let $p,q$ be multivariate probability density functions with independent components, ie,

$$
p(x)=\prod_{i=1}^k p_i(x_i)\quad\text{ and }\quad q(x)=\prod_{i=1}^kq_i(x_i),
$$
 
where $p_i,q_i$ are univariate probability density functions. 
Then 

$$
\KL(p||q)=\sum_{i=1}^k\KL(p_i||q_i)
$$
```

````{prf:Proof}

We start with:
\begin{align*}
\KL(p||q) & = \expec_p[\log p-\log q] \\
& = \expec_p\left[\log\left(\prod_{i=1}^kp_i(x_i)\right) - \log\left(\prod_{i=1}^kq_i(x_i)\right) \right] \\
& = \expec_p\left[\sum_{i=1}^k\log\left(p_i(x_i)\right) - \log\left(q_i(x_i)\right) \right] \\
& = \sum_{i=1}^k \expec_p\left[\log\left(p_i(x_i)\right) - \log\left(q_i(x_i)\right)\right] \\
& = \sum_{i=1}^k \expec_{p_1}\ldots\expec_{p_k}\left[\log\left(p_i(x_i)\right) - \log\left(q_i(x_i)\right)\right] \\
& = \sum_{i=1}^k \expec_{p_i}\left[\log\left(p_i(x_i)\right) - \log\left(q_i(x_i)\right)\right], 
\end{align*}
as neither $\log p_i$ nor $\log q_i$ depends on $x_j$ for $i\neq j$.

To finish the proof we observe that 

$$
\expec_{p_i}\left[\log\left(p_i(x_i)\right) - \log\left(q_i(x_i)\right)\right] = \KL(p_i||q_i)
$$
````

We also need
the first and second moments of a univariate normal distribution.
These are [well-known:](https://en.wikipedia.org/wiki/Normal_distribution)

$$
\expec_{\mathcal N(\mu,\sigma^2)}[x]=\mu\quad\text{ and }
\expec_{\mathcal N(\mu,\sigma^2)}[x^2]=\mu^2+\sigma^2
$$

We now prove {prf:ref}`KLnormalthm`
````{prf:Proof}

By {prf:ref}`multiKLlem` it suffices to prove the theorem in the univariate case, ie, when $k=1$.

We start with 
\begin{align*}
\KL(\mathcal N(\mu,\sigma^2)||\mathcal N(0,1))  = & \expec_{\mathcal N(\mu,\sigma^2)}\left[
\log\left(\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}\right)
\right]\\
&-\expec_{\mathcal N(\mu,\sigma^2)}\left[
\log\left(\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}\right)
\right]\\
\end{align*}
and then look at the two terms separately.

The first term becomes
\begin{align*}
\expec_{\mathcal N(\mu,\sigma^2)}&\left[
\log\left(\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}\right)
\right] \\
& = -\tfrac{1}{2}\log\left(2\pi\sigma^2\right) - \expec_{\mathcal N(\mu,\sigma^2)}\left[\frac{(x-\mu)^2}{2\sigma^2}\right]\\
& = -\tfrac{1}{2}\log\left(2\pi\sigma^2\right) - \frac{1}{2\sigma^2}\expec_{\mathcal N(\mu,\sigma^2)}[x^2]
+\frac{2\mu}{2\sigma^2}\expec_{\mathcal N(\mu,\sigma^2)}[x] - \frac{\mu^2}{2\sigma^2} \\
& = -\tfrac{1}{2}\log\left(2\pi\sigma^2\right) - \frac{\mu^2+\sigma^2}{2\sigma^2}
+\frac{2\mu^2}{2\sigma^2} - \frac{\mu^2}{2\sigma^2}
=-\tfrac{1}{2}\log\left(2\pi\sigma^2\right)  -\tfrac{1}{2}
\end{align*}
The second term becomes
\begin{align*}
\expec_{\mathcal N(\mu,\sigma^2)}&\left[
\log\left(\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}\right)
\right]\\
& = -\tfrac{1}{2}\log(2\pi) -\tfrac{1}{2}\expec_{\mathcal N(\mu,\sigma^2)}[x^2] \\
& = -\frac{1}{2}\log(2\pi)-\frac{\mu^2+\sigma^2}{2}
\end{align*}

Putting the terms together, we obtain
\begin{align*}
\KL&(\mathcal N(\mu,\sigma^2)||\mathcal N(0,1)) \\
& \quad = -\frac{1}{2}\log\left(2\pi\sigma^2\right)  -\frac{1}{2} +\frac{1}{2}\log(2\pi)+\frac{\mu^2+\sigma^2}{2}\\
&\quad = \tfrac{1}{2}\left(-\log\sigma^2-1+\mu^2+\sigma^2\right),
\end{align*}
which finishes the proof.
````




