Stochastic gradient descent
===========================

How is a neural network trained? How can we minimise logistic loss in
order to learn the parameters of a logistic regression? Both cases
reduce to an optimisation problem that requires a numerical optimisation
algorithm, often a variant of a gradient descent technique. In the
nicest and simplest setting, a convex optimisation problem, these are
even guaranteed to find an optimal solution.

Convexity
---------

A set :math:`C\subseteq\mathbb R^n` is :index:`\ <convex set>`\ *convex*
if the connecting segment between
any two points in :math:`C` is also contained in :math:`C`:

.. math:: \lambda x+(1-\lambda)y\in C\text{ for all }x,y\in C\text{ and }\lambda\in[0,1].

.. _convsetfig:
.. figure:: pix/convexsets.png
    :figwidth: 12 cm
    
    A convex set in  (a) and (b); the set in (c) is not convex

Let :math:`C\subseteq\mathbb R^n` be a convex set. A function
:math:`f:C\to\mathbb R` is a :index:`\ <convex function>`\ *convex function* if for all :math:`x,y\in C` and all
:math:`\lambda\in[0,1]` it holds that

.. math:: 
    :label: convdef
    
    f(\lambda x+(1-\lambda) y)\leq \lambda f(x) + (1-\lambda) f(y)

Obviously, a linear function is always convex.

.. _convfunfig:
.. figure:: pix/convexconcave.png
    :figwidth: 12 cm
    
    A convex function, a concave function (negative of a convex function), and a function that is neither convex nor concave.

The two notions of convexity, convex sets and convex functions, are
related via the epigraph of a function. For a function
:math:`f:C\to\mathbb R`, the :dfn:`epigraph` is defined as

.. math::
   
   \text{epi}(f)=\{(x,y) : x\in C,y\geq f(x)\}

That means, the epigraph is simply the set of all points above the
function graph.

Convex sets and convex functions are now related as follows:

.. prf:Proposition:: 

   Let :math:`C\subseteq\mathbb R^n` be a convex set, and let
   :math:`f:C\to\mathbb R` be a function. Then :math:`f` is a convex
   function if and only if the epigraph :math:`\text{epi}(f)` is a
   convex set.


Convex optimisation problems
----------------------------

A :dfn:`convex optimisation problem` is any problem of the form

.. math::
   :label: convopt

   \inf f(x),\quad x\in K

where :math:`K\subseteq \mathbb R^n` is a convex set and
:math:`f:K\to\mathbb R` a convex function.

A point :math:`x^*\in K` is a :dfn:`local minimum` if there is an open ball :math:`B` around
:math:`x^*` such that

.. math:: f(x^*)\leq f(x) \text{ for all }x\in B\cap K

.. prf:Proposition::

   If :math:`x^*` is a local minimum of :eq:`convopt` then it
   is also a global minimum.


.. prf:Proof::

   Suppose there is a :math:`z\in K` with :math:`f(z)<f(x^*)`.
   Let :math:`B` be a ball around :math:`x^*` such that
   :math:`f(x^*)\leq f(x)` for all :math:`x\in B\cap K`. Since :math:`K`
   is convex, :math:`x_\lambda=\lambda x^*+(1-\lambda)z\in K` for all
   :math:`\lambda\in [0,1]`. In particular, there is a
   :math:`\lambda\in (0,1]` such that :math:`x_\lambda\in B`. Because
   :math:`f` is convex

   .. math:: f(x_\lambda)\leq \lambda f(x^*)+(1-\lambda)f(z)<f(x^*)

   as :math:`\lambda\neq 0` and :math:`f(z)<f(x^*)`. This, however, is a
   contradiction to :math:`x^*` being a local minimum. ◻

Note that it makes a difference whether we aim to minimise or maximise a
convex function over a convex set. Indeed, if we maximise the function
in Figure :numref:`convfunfig` over the convex set
:math:`[0,1]` we see that :math:`x^*=0` is a local maximum but not a
global one (that would be :math:`z=1`).

Convex functions
----------------

Which functions are convex? Norms are convex. Indeed, the function
:math:`x\mapsto ||x||` is convex as for every :math:`\lambda\in [0,1]`
the triangle inequality implies:

.. math:: ||\lambda x+(1-\lambda) y||\leq ||\lambda x|| + ||(1-\lambda) y|| = \lambda||x||+(1-\lambda)||y||

Recall that
:math:`\nabla f(x) = \trsp{\left(\frac{\partial f}{\partial x_1}(x),\ldots,\frac{\partial f}{\partial x_n}(x)\right)}`
is the :dfn:`gradient` of :math:`f` at :math:`x`.

.. _gradlem:
.. prf:Lemma::

   Let :math:`f:C\to\mathbb R` be a differentiable function on
   an open convex set :math:`C\subseteq \mathbb R^n`. Then :math:`f` is
   convex if and only if

   .. math:: f(y)\geq f(x)+\trsp{\nabla f(x)}(y-x)\text{ for all }x,y\in C.

.. prf:Proof::

   First we do :math:`n=1`, i.e. we prove that

   .. math:: \emtext{$f$ is convex} \quad\Leftrightarrow\quad  f(y)\geq f(x)+f'(x)(y-x)\emtext{ for all }x,y\in C

   Assume first that :math:`f` is convex. Then for every
   :math:`\lambda\in[0,1]`

   .. math::

      \begin{aligned}
      \lambda f(y) &\geq f(x+\lambda(y-x))-(1-\lambda) f(x)\end{aligned}

   We divide by :math:`\lambda`:

   .. math::

      \begin{aligned}
      f(y) &\geq \frac{f(x+\lambda(y-x))-f(x)}{\lambda}+f(x)\\
      &=\frac{f(x+\lambda(y-x))-f(x)}{\lambda(y-x)}(y-x)+f(x)\\
      &= \frac{f(x+t)-f(x)}{t}(y-x)+f(x)\end{aligned}

   for :math:`t=\lambda(y-x)`. Now taking :math:`t\to 0`, we get
   :math:`f(y)\geq f(x)+f'(x)(y-x)`.

   For the other direction, we put :math:`z=\lambda x+(1-\lambda)y`, and
   obtain

   .. math:: f(x)\geq f(z)+f'(z)(x-z)\emtext{ and }f(y)\geq f(z)+f'(z)(y-z)

   We multiply the first inequality with :math:`\lambda`, the second
   with :math:`(1-\lambda)` and add them. This finishes the case
   :math:`n=1`.

   For :math:`n>1`, we define :math:`g:[0,1]\to\mathbb R` by
   :math:`g(\lambda)=f(\lambda x+(1-\lambda) y)` and then apply the
   one-dimensional case. We omit the details. ◻

If a function is twice differentiable then whether it is convex can be
read off its second derivative:

.. _twicelem:
.. prf:Lemma::

   Let :math:`f:C\to\mathbb R` be a twice differentiable
   function on an open interval :math:`C\subseteq \mathbb R`. Then the
   following statements are equivalent:

   #. :math:`f` is convex;

   #. :math:`f'` is monotonically non-decreasing; and

   #. :math:`f''` is non-negative.

Again, I omit the proof. There is also a version for multivariate
functions.

As a consequence of the lemma, :math:`x\mapsto x^2` is a convex function
over :math:`\mathbb R`, and so is :math:`x\mapsto e^x`. Also, the
function :math:`f:x\mapsto \log(1+e^x)` is convex: Indeed,

.. math:: f'(x)=\frac{e^x}{1+e^x}=\frac{1}{1+e^{-x}},

which is monotonically increasing.

Compositions of convex functions are not generally convex: Indeed, both
:math:`f:x\mapsto x^2` and :math:`g:x\mapsto e^{-x}` are convex,
but :math:`g\circ f:x\mapsto e^{-x^2}` is not. This is different if the
inner function is affine.

.. _affconvlem:
.. prf:Lemma::

   Let :math:`g:\mathbb R\to\mathbb R` be convex, and let
   :math:`w\in\mathbb R^n` and :math:`b\in\mathbb R`. Then
   :math:`f(x)=g(\trsp wx+b)` is also convex.

.. prf:Proof::

   Let :math:`x,y\in\mathbb R^n` and :math:`\lambda\in [0,1]`.
   Then

   .. math::

      \begin{aligned}
      f(\lambda x+(1-\lambda)y) &= g(\lambda (\trsp wx+b) + (1-\lambda)(\trsp wy+b))\\
      &\leq \lambda g(\trsp wx+b) + (1-\lambda) g(\trsp wy+b)\\
      & = \lambda f(x)+(1-\lambda)f(y),\end{aligned}

   as :math:`g` is convex. ◻

As a consequence, for fixed :math:`x\in\mathbb R^n`,
:math:`y\in\mathbb R` the function :math:`f:\mathbb R^n\to\mathbb R`,
:math:`w\mapsto \log(1+e^{-y\trsp wx})` is convex.

The following statement is almost trivial to prove:

.. _sumlem:
.. prf:Lemma::

   Let :math:`C\subseteq\mathbb R^n` be a convex set, let
   :math:`w_1,\ldots, w_m\geq 0`, and let
   :math:`f_1,\ldots,f_m:C\to\mathbb R` be convex functions. Then
   :math:`f=\sum_{i=1}^mw_if_i` is a convex function.

Recall that logistic regression works by minimising the logistic loss.
As a consequence of the previous lemmas, we get:

.. _loglosslem:
.. prf:Lemma::

   For every finite training set
   :math:`S\subseteq \mathbb R^n\times\{-1,1\}`, the logistic loss
   function

   .. math:: w\mapsto \frac{1}{|S|}\sum_{(x,y)\in S}-\log_2\left(\sigm(y\trsp wx)\right)

   is convex.

Recall that

.. math:: \sigm:z\mapsto \frac{1}{1+e^{-z}}

.. prf:Proof::

   We have already seen that the function
   :math:`f:\mathbb R^n\to\mathbb R`,
   :math:`w\mapsto \log(1+e^{-y\trsp wx})` is convex, for fixed
   :math:`x\in\mathbb R^n` and :math:`y\in\{-1,1\}`. Now, the logistic
   loss is simply the sum of such functions, weighted with the positive
   factor :math:`\tfrac{1}{|S|}`:

   .. math::

      \frac{1}{|S|}\sum_{(x,y)\in S}-\log_2\left(\sigm(y\trsp wx)\right)
      = \frac{1}{|S|}\sum_{(x,y)\in S}\log_2\left(1+e^{-y\trsp wx}\right)

   Thus, it follows from Lemma :numref:`sumlem` that the logistic
   loss function is convex. ◻

Recall that when performing logistic regression we aim to find a linear
classifier with small zero-one loss. Instead of minimising the zero-one
loss directly, however, we minimise the logistic loss – which we had
seen to upper-bound the zero-one loss; see
Lemma :numref:`upperloglosslem`. Here now is the reason,
why we replace the zero-one loss by a function, the logistic loss: in
contrast to zero-one loss, the logistic loss function is convex!

Let’s look at one more way to obtain a convex function.

.. _suplem:
.. prf:Lemma::

   Let :math:`I` be some index set, let :math:`C` be a convex
   set. Let :math:`f_i:C\to\mathbb R`, :math:`i\in I`, be a family of
   convex functions. Then :math:`f:x\mapsto\sup_{i\in I}f_i(x)` is a
   convex function.

.. prf:Proof::

   Let :math:`x,y\in C` and :math:`\lambda\in [0,1]`. Then for
   every :math:`i^*\in I`, because :math:`f_{i^*}` is convex, it holds
   that:

   .. math::

      \begin{aligned}
      f_{i^*}(\lambda x+(1-\lambda)y) &\leq \lambda f_{i^*} + (1-\lambda) f_{i^*}(y)\\
      & \leq \sup_{i\in I}\lambda f_{i} + (1-\lambda) f_{i}(y)
      \leq \lambda \sup_{i\in I}f_{i} + (1-\lambda) \sup_{i\in I}f_{i}(y)\end{aligned}

   Therefore it also holds that

   .. math::

      \sup_{i\in I}f_{i}(\lambda x+(1-\lambda)y)
      \leq \lambda \sup_{i\in I}f_{i} + (1-\lambda) \sup_{i\in I}f_{i}(y)

    ◻

Strong convexity
----------------

Many of the functions we encounter in machine learning are at least
locally convex, and usually these even exhibit a stronger notion of
convexity that is called, well, *strong* convexity. The difference
between convexity and strong convexity is basically the difference
between an affine function such as :math:`x\mapsto x` and a quadratic
function such as :math:`x\mapsto x^2`. Affine functions are convex but
barely so: they satisfy the defining inequality of
convexity :eq:`convdef` with equality. For a strongly convex
function this will never be the case.

A function :math:`f:K\to\mathbb R` on a convex set
:math:`K\subseteq\mathbb R^d` is :dfn:`*mu*-strongly convex` for :math:`\mu>0` if for all
:math:`\lambda\in [0,1]` and :math:`x,y\in K` it holds that

.. math::

   \lambda f(x)+ (1-\lambda)f(y)\geq f(\lambda x+(1-\lambda)y) +\frac{\mu}{2}\lambda(1-\lambda)
   ||x-y||^2_2

Clearly, it is the additional term
:math:`\frac{\mu}{2}\lambda(1-\lambda)
||x-y||^2_2` that makes strong convexity a stronger notion than ordinary
convexity. In particular, affine functions are convex but not
:math:`\mu`-strongly convex for any :math:`\mu>0`.

.. _strongnormlem:
.. prf:Lemma::

   The function :math:`\mathbb R^d\to\mathbb R`,
   :math:`x\mapsto ||x||^2_2` is :math:`2`-strongly convex.

.. prf:Proof::

   Let :math:`\lambda\in[0,1]` and :math:`x,y\in\mathbb R^d`.
   Then

   .. math:: ||\lambda x+(1-\lambda)y||^2 = \lambda^2||x||^2+2\lambda(1-\lambda)\trsp xy+(1-\lambda)^2||y||^2

   and

   .. math:: \lambda(1-\lambda)||x-y||^2 = \lambda(1-\lambda)||x||^2-2\lambda(1-\lambda)\trsp xy+\lambda(1-\lambda)||y||^2

   Adding the two right-hand sides gives
   :math:`\lambda ||x||^2+(1-\lambda)||y||^2`. ◻

.. _stronglem:
.. prf:Lemma::

   Let :math:`g:K\to\mathbb R` be a :math:`\mu`-strongly
   convex function on a convex set :math:`K\subseteq\mathbb R^d`. Then

   #. :math:`Cg` is :math:`C\mu`-strongly convex for any :math:`C>0`;
      and

   #. if :math:`f:K\to\mathbb R` is convex then :math:`f+g` is
      :math:`\mu`-strongly convex.

.. prf:Proof::

   (i) is trivial and so is (ii). ◻

Here, statement (ii) is the reason why strong convexity is relevant to
us. Often, we might have a convex loss function :math:`L(w)` and then
add a term :math:`\mu||w||_2` to the loss function that penalises large
weights. This is a common strategy, called *regularisation*, that we
will treat later. A fortunate consequence is then that the new function
:math:`w\mapsto L(w)+\mu||w||_2^2` is even strongly convex.

.. _strongdifflem:
.. prf:Lemma::

   Let :math:`f:K\to\mathbb R` be a differentiable
   function on an open convex set :math:`K\subseteq\mathbb R^d`. Then
   :math:`f` is :math:`\mu`-strongly convex if and only if for all
   :math:`x,y\in K`

   .. math:: f(y)\geq f(x)+\nabla \trsp{f(x)} (y-x)+\frac{\mu}{2}||y-x||^2

The proof is an obvious modification of the proof of
Lemma :numref:`gradlem`.

We draw a simple consequence. If :math:`x` is a global minimum of
:math:`f` then, as :math:`\nabla f(x)=0` it follows that

.. math::
   :label: strongmin2
   
   f(y)-f(x)\geq \frac{\mu}{2}||y-x||^2


Gradient descent
----------------

Some of the objective functions in machine learning are convex. How can
we minimise them? With *stochastic gradient descent* – it is this
algorithm (or one of its variants) that powers most of machine learning.
Let’s understand simple :dfn:`gradient descent` first.

.. prf:algorithm::

    **Instance** A differentiable function :math:`f:\mathbb R^n\to\mathbb R`, a first point :math:`x^{(1)}`.
    
    **Output** A point :math:`x`.
    
    1. Set :math:`t=1`. 
    2. While stopping criterion not satisfied:
        3. Compute :math:`\nabla f(x^{(t)})`.
        4. Compute learning rate :math:`\eta_t`.
        5. Set :math:`x^{(t+1)}=x^{(t)}-\eta_t\nabla f(x^{(t)})`.
        6. Set :math:`t=t+1`.
    7. Output :math:`x^{(t)}`, or best of :math:`x^{(1)},\ldots, x^{(t)}`, or average.
   
[#mn1]_ There are different strategies for the learning rate :math:`\eta_t`
(which should always be positive). The easiest is a constant learning
rate :math:`\eta_t=\eta>0` for all :math:`t`. The problem here is that
at the beginning of gradient descent, a constant learning rate will
probably lead to slow progress, while near the minimum, it might lead to
overshooting. More common are decreasing or adaptive learning rates, see
below.

.. [#mn1] {-} `gradient descent <https://colab.research.google.com/github/henningbruhn/math_of_ml_course/blob/main/stochastic_gradient_descent/gradient.ipynb>`_ 

Typical stopping criteria are: a pre-fixed maximum number of iterations
has been reached; or the norm of the gradient has become very small.

Concerning the output: rather than outputting the last :math:`x^{(t)}`
it seems that it cannot hurt to output the best :math:`x^{(t)}`
encountered during the execution – that, however, necessitates a
function evaluation :math:`f(x^{(t)})` in every step, which can be
computationally costly. From a theoretical point of view, the average
:math:`\tfrac{1}{T}\sum_{t=1}^Tx^{(t)}` is sometimes convenient.

.. figure:: pix/gd_etas.png
   :name: etafig
   :width: 15cm

   Gradient descent with constant learning rates of different values.
   The function to be minimised is
   :math:`(x,y)\mapsto \tfrac{1}{2}(x^2+10y^2)`. Middle: small learning
   rate leads to slow convergence. Right: learning rate is too large, no
   convergence.

.. card:: 
   :class-body: div.theorem

   Theorem X
   ^^^

   If :math:`f:\mathbb R^n\to \mathbb R` is a convex and differentiable
   function and certain additional but mild conditions are satisfied, in
   particular with respect to the learning rate, then will converge
   towards the global minimum:

   .. math:: x^{(t)}\to x^*,\text{ as }t\to\infty,

   where :math:`x^*` is a global minimum of :math:`f`.

.. prf:Theorem::

   If :math:`f:\mathbb R^n\to \mathbb R` is a convex and differentiable
   function and certain additional but mild conditions are satisfied, in
   particular with respect to the learning rate, then will converge
   towards the global minimum:

   .. math:: x^{(t)}\to x^*,\text{ as }t\to\infty,

   where :math:`x^*` is a global minimum of :math:`f`.

The statement is intentionally vague. Indeed, there are a number of such
results, each with its own set of specific conditions. The main point
is: For a convex function gradient descent will normally converge. We
will not discuss this in more detail as plain gradient descent is almost
never used in machine learning.

.. dropdown:: Gradient descent – an old technique
   :color: success
   :icon: rocket

   .. image:: pix/cauchy_methode_generale.png
        :width: 10cm
        :align: left

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


.. _sgdsec:

Stochastic gradient descent
---------------------------

Gradient descent is a quite efficient algorithm. Under mild assumptions
and with the right (adaptable) learning rate it can be shown that the
error :math:`\epsilon`, the difference :math:`f(\overline x)-f(x^*)`,
decreases exponentially with the number of iterations, i.e. that

.. math:: \log(1/\epsilon) \sim t \leftrightarrow \epsilon \sim e^{-t}

Why is not normally used in machine learning? Let’s consider logistic
regression, where we have the logistic loss function

.. math:: w\mapsto \frac{1}{|S|}\sum_{(x,y)\in S}\log_2\left(1+e^{-y\trsp wx}\right)

.. code:: python
   :number-lines:

   def my_function():
       "just a test"
       print(8/2)

