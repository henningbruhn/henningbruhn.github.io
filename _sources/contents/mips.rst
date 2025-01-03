Maximum inner product search
============================

.. see https://en.wikipedia.org/wiki/Vector_database

:index:`\ <vector database>`\ *Vector databases* are becoming more and more important. 
What's a vector database? A system to store a large number of vectors 
:math:`x^{(1)},\ldots, x^{(n)}\in\mathbb R^d`
in such a way that a (approximate) nearest neighbour search can be performed efficiently.    
A recommender system, for instance, might
store the preferences of the users encoded as vectors; for a new user the five most similar
known users could be computed in order to recommend the products or services they prefered. 
Another application comes from word or document embeddings: A number of vector representation 
of documents are stored in the database; a user may then formulate a query ("which Tom Stoppard play 
features Hamlet as a side character?") that is transformed into a vector; the documents with 
most similar vector representation are then returned.

What :emphasis:`most similar` means will differ from application to application. Often it may
simply mean: the largest scalar product. That is, given a query 
:math:`q\in\mathbb R^d` we look for the 
:math:`x^{(i)}`
with largest 
:math:`q^\intercal x^{(i)}`. 
In that case, the problem is known as :index:`\ <maximum inner product search>`\ *maximum inner product search* (or MIPS).

At first, the computational problem may appear to have an easy solution -- after all, scalar 
products can be computed very efficiently. With :math:`n` vectors in the database, each with length :math:`d`,
checking every scalar product amounts to a running time of :math:`O(nd)`. The number of vectors, :math:`n`, 
however, will usually be extremely large, perhaps even in the billions, and the dimension :math:`d` may 
have three or four digits. Moreover, queries typically need to be answered very quickly. A recommender system, 
for example, could easily need to address hundreds or thousands of queries every second. 
As a result, a running time of :math:`O(nd)` may be too slow. 
How can this be sped up?

.. dropdown:: Retrieval augmented generation
    :color: success
    :icon: rocket

    It is not easy to keep an AI chatbot up-to-date with current affairs. 
    The large language model (LLM) it is based on is trained on a snapshot of data 
    available at training time (perhaps all of wikipedia as of 07/10/24). Later
    events (announcement of the Physics Nobel Prize on 08/10/24) are thus not immediately accessible 
    to the bot. 

    Re-training an LLM to incorporate later events is resource intensive and therefore only 
    done rarely. An alternative is to provide the AI chatbot with an external memory. One such 
    method is :index:`\ <retrieval augmented generation>`\ *retrieval augmented generation*, or RAG. 

    In RAG, the user query is transformed into a vector via a topic embedding that is compared to 
    data in a vector database (this could be a current snapshot of wikipedia). The perhaps ten documents
    that are most similar to the query are located and added to the prompt of the AI chatbot. The
    bot then generates the response based on the query and the search results. 

    *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 
    P. Lewis et al. (2021), `arXiv:2005.11401 <https://arxiv.org/pdf/2005.11401>`_ 

Vector quantisation
-------------------

.. this is largely from
.. Quantization based Fast Inner Product Search
.. Ruiqi Guo, Sanjiv Kumar, Krzysztof Choromanski, David Simcha
.. arXiv:1509.01469

Let :math:`x^{(1)},\ldots x^{(n)}\in\mathbb R^d` be the vectors  that make up the database,
let :math:`k,m>0` be integers, and let :math:`\ell=\tfrac{d}{m}`, which we assume to be an integer. [#fQuant]_

We split each vector :math:`x\in\mathbb R^d` into :math:`m` vectors each of length :math:`\ell`:

.. math::
    x=\begin{pmatrix}x_1\\x_2\\\vdots\\x_m\end{pmatrix}, \text{ with }x_j\in\mathbb R^\ell\text{ for }j=1,\ldots, m

(In practice, this partition of :math:`\mathbb R^d` into subspaces is achieved by random sampling of the entry dimensions.)

Then, for :math:`j=1,\ldots,m` we compute representative vectors :math:`c_{1j},\ldots,c_{kj}\in\mathbb R^\ell`.
These are used to replace the database vectors by simpler vectors: For each :math:`i=1,\ldots, n` and :math:`j=1,\ldots, m` 
we find a suitable :math:`\hat x^{(i)}_j\in\{c_{j1},\ldots,c_{jk}\}`. That is, we basically replace :math:`x^{(i)}_j` by one of :math:`\{c_{j1},\ldots,c_{jk}\}`.
In this way :math:`x^{(i)}` is replaced by 

.. math::
    \hat x^{(i)}=\begin{pmatrix}\hat x^{(i)}_1\\\vdots\\\hat x^{(i)}_m\end{pmatrix}

For a query :math:`q\in\mathbb R^d` we then approximate :math:`q^\intercal x^{(i)}\approx q^\intercal\hat x^{(i)}`. 

Before we look at :math:`q\hat x^{(i)}` let me point out that replacing each :math:`x^{(i)}` by :math:`\hat x^{(i)}` 
already results in a welcome compression of the data. Indeed, we only need to store all :math:`c_{js}`, :math:`j=1,\ldots, m`, :math:`s=1,\ldots, k`
and, instead of :math:`x^{(i)}` we store a vector :math:`\hat z^{(i)}\in\{1,\ldots,k\}^m` with :math:`\hat z^{(i)}_j=s` if and only if :math:`\hat x^{(i)}_j=c_{js}`.

What is the computational complexity to compute :math:`q^\intercal\hat x^{(i)}` for all :math:`i`?
We write

.. math::
    q^\intercal\hat x^{(i)} = \sum_{j=1}^mq^\intercal_j\hat x^{(i)}_j

Note that :math:`\hat x^{(i)}_j` is one of :math:`c_{j1},\ldots, c_{jk}`. We first compute all scalar products :math:`q^\intercal_jc_{js}` and 
put them in a look-up table. Computing the scalar product takes :math:`O(mk\ell)` time as :math:`j` runs from :math:`1` to :math:`m`, :math:`s` runs
from :math:`1` to :math:`k` and as the vectors have length :math:`\ell`. With :math:`\ell=\tfrac{d}{m}` the running time reduces to :math:`O(kd)`.
Then, using the look-up table, we compute for all :math:`i` the scalar product :math:`q^\intercal\hat x^{(i)}`, with a running time of 
:math:`O(nm)`. In total we obtain a running time of :math:`O(kd+nm)`. As :math:`n` is typically the by far largest quantity among
:math:`d,k,n,m` the running time is dominated by :math:`O(nm)`, and as usually :math:`m \ll d`, we achieve a substantial reduction 
in comparison to the running time :math:`O(nd)` of the naive algorithm.

.. sidebar:: Retrieval augmented generation

    It is not easy to keep an AI chatbot up-to-date with current affairs. 
    The large language model (LLM) it is based on is trained on a snapshot of data 
    available at training time (perhaps all of wikipedia as of 07/10/24). Later
    events (announcement of the Physics Nobel Prize on 08/10/24) are thus not immediately accessible 
    to the bot. 

    Re-training an LLM to incorporate later events is resource intensive and therefore only 
    done rarely. An alternative is to provide the AI chatbot with an external memory. One such 
    method is :index:`\ <retrieval augmented generation>`\ *retrieval augmented generation*, or RAG. 

    In RAG, the user query is transformed into a vector via a topic embedding that is compared to 
    data in a vector database (this could be a current snapshot of wikipedia). The perhaps ten documents
    that are most similar to the query are located and added to the prompt of the AI chatbot. The
    bot then generates the response based on the query and the search results. 

    *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 
    P. Lewis et al. (2021), `arXiv:2005.11401 <https://arxiv.org/pdf/2005.11401>`_ 


How should the representatives :math:`c_{ji},\ldots,c_{jk}\in\mathbb R^\ell` be chosen? Well, :math:`q^\intercal\hat x^{(i)}`
should be a good approximation of :math:`q^\intercal x^{(i)}`, which is the case if :math:`q^\intercal_j\hat x^{(i)}_j` is a 
good approximation of :math:`q^\intercal_jx^{(i)}_j` for every :math:`j\in\{1,\ldots, m\}`. So, let's fix :math:`j` and 
let us assume that the queries :math:`q` are drawn from some distribution :math:`\mathcal D`. 
Then, arguably, we should choose :math:`c_{j1},\ldots,c_{jk}` such that 

.. math::
    :label: vqobj
    
    \mathbb E_{q\sim\mathcal D}\left[\sum_{i=1}^n(q^\intercal_jx^{(i)}_j-q^\intercal_j\hat x^{(i)}_j)^2\right]

is minimised. Let us rewrite that.

.. math::

    \mathbb E_{q\sim\mathcal D}\left[\sum_{i=1}^n\left(q^\intercal_jx^{(i)}_j-q^\intercal_j\hat x^{(i)}_j\right)^2\right]
    &= \sum_{i=1}^n\mathbb E_{q\sim\mathcal D}\left[\left(q^\intercal_j\left(x^{(i)}_j-\hat x^{(i)}_j\right)\right)^2\right]\\
    &=\sum_{i=1}^n\mathbb E_{q\sim\mathcal D}\left[\left( \cos\theta_{ij} ||q_j||\cdot ||x^{(i)}_j-\hat x^{(i)}_j|| \right)^2\right]\\
    &=\sum_{i=1}^n||x^{(i)}_j-\hat x^{(i)}_j||^2\cdot\mathbb E_{q\sim\mathcal D}\left[\left( \cos\theta_{ij} ||q_j||\right)^2\right],

where we write :math:`\theta_{ij}` for the angle between :math:`q_j` and :math:`x^{(i)}_j-\hat x^{(i)}_j`. 


We've reached a point where we are stuck without further assumption on the distribution :math:`\mathcal D`. 
We do not want to impose strong conditions on it as it is quite outside our control. 
However, not too far of a stretch seems to assume that :math:`\mathcal D` is :emphasis:`isotropic`, ie, does not 
depend on the direction. Under that assumption, we obtain:

.. math::

    \mathbb E_{q\sim\mathcal D}\left[\sum_{i=1}^n\left(q^\intercal_jx^{(i)}_j-q^\intercal_j\hat x^{(i)}_j\right)^2\right]
    =C\sum_{i=1}^n||x^{(i)}_j-\hat x^{(i)}_j||^2,

for some constant :math:`C>0` that only depends on :math:`\mathcal D` but not on :math:`\hat x^{(i)}_j`.

Recall that each :math:`\hat x^{(i)}_j` is one of :math:`c_{j1},\ldots,c_{jk}`. Thus 

.. math::

    \text{argmin}_{c_{j1},\ldots,c_{jk}} \mathbb E_{q\sim\mathcal D}\left[\sum_{i=1}^n\left(q^\intercal_jx^{(i)}_j-q^\intercal_j\hat x^{(i)}_j\right)^2\right]
    = \text{argmin}_{c_{j1},\ldots,c_{jk}} \sum_{i=1}^n\min_{s}||x^{(i)}_j-c_{js}||^2,

which is nothing else than the :math:`k`-means objective! 

What does that mean? We split each database vector :math:`x^{(i)}` into chunks of size :math:`\tfrac{d}{m}`, and then, for each :math:`j=1,\ldots, m`, 
solve for the data :math:`x^{(1)}_j,\ldots, x^{(n)}_j` a :math:`k`-means clustering problem, resulting 
in the centres :math:`c_{j1},\ldots,c_{jk}`; finally we replace each :math:`x^{(i)}_j` with the nearest cluster centre :math:`c_{js}`.

.. index:: vector quantisation

Replacing vectors in a dataset by simpler vectors is  called :emphasis:`vector quantisation`, and also used as a compression tool, 
for example, in video or audio codecs. 

|

The approach outlined here can be improved upon. Indeed, I've cheated with the objective :eq:`vqobj`: The objective
incorporates :emphasis:`all` scalar products :math:`q^\intercal\hat x^{(i)}`. In the applications, however, we just want to find
the datapoint with largest scalar product with the query, or rather, the top-10 datapoints with largest scalar product. 
Thus, it does not matter if :math:`q^\intercal\hat x^{(i)}` deviates from :math:`q^\intercal x^{(i)}` as long as both scalar products 
are not too large. This insight can be used to devise a more complicated loss function that results in 
better performance; see Guo et al. (2020). [#fGuo]_

.. see wikipedia... vector quantization 


.. rubric:: Footnotes

.. [#fQuant] :emphasis:`Quantization based Fast Inner Product Search`, R. Guo, S. Kumar, K. Choromanski and D. Simcha (2015), `arXiv:1509.01469 <https://arxiv.org/abs/1509.01469>`_ 

.. [#fGuo] :emphasis:`Accelerating Large-Scale Inference with Anisotropic Vector Quantization`, R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern and S. Kumar (2020), `arXiv:1908.10396 <https://arxiv.org/pdf/1908.10396>`_



