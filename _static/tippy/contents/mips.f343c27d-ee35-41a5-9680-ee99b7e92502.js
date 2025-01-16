selector_to_html = {"a[href=\"#vector-quantisation\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">8.1. </span>Vector quantisation<a class=\"headerlink\" href=\"#vector-quantisation\" title=\"Link to this heading\">#</a></h2><p>Let <span class=\"math notranslate nohighlight\">\\(x^{(1)},\\ldots x^{(n)}\\in\\mathbb R^d\\)</span> be the vectors  that make up the database,\nlet <span class=\"math notranslate nohighlight\">\\(k,m&gt;0\\)</span> be integers, and let <span class=\"math notranslate nohighlight\">\\(\\ell=\\tfrac{d}{m}\\)</span>, which we assume to be an integer.<label class=\"margin-toggle\" for=\"sidenote-role-1\"><span id=\"id1\">\n<sup>1</sup></span>\n</label><input class=\"margin-toggle\" id=\"sidenote-role-1\" name=\"sidenote-role-1\" type=\"checkbox\"/><span class=\"sidenote\"><sup>1</sup><em>Quantization based Fast Inner Product Search</em>, R. Guo, S. Kumar, K. Choromanski and D. Simcha (2015), <a class=\"reference external\" href=\"https://arxiv.org/abs/1509.01469\">arXiv:1509.01469</a></span></p><p>We split each vector <span class=\"math notranslate nohighlight\">\\(x\\in\\mathbb R^d\\)</span> into <span class=\"math notranslate nohighlight\">\\(m\\)</span> vectors each of length <span class=\"math notranslate nohighlight\">\\(\\ell\\)</span>:</p>", "a[href=\"#maximum-inner-product-search\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">8. </span>Maximum inner product search<a class=\"headerlink\" href=\"#maximum-inner-product-search\" title=\"Link to this heading\">#</a></h1><p><em>Vector databases</em> are becoming more and more important.\nWhat\u2019s a vector database? A system to store a large number of vectors\n<span class=\"math notranslate nohighlight\">\\(x^{(1)},\\ldots, x^{(n)}\\in\\mathbb R^d\\)</span>\nin such a way that a (approximate) nearest neighbour search can be performed efficiently.<br/>\nA recommender system, for instance, might\nstore the preferences of the users encoded as vectors; for a new user the five most similar\nknown users could be computed in order to recommend the products or services they prefered.\nAnother application comes from word or document embeddings: A number of vector representation\nof documents are stored in the database; a user may then formulate a query (\u201cwhich Tom Stoppard play\nfeatures Hamlet as a side character?\u201d) that is transformed into a vector; the documents with\nmost similar vector representation are then returned.</p><p>What <em>most similar</em> means will differ from application to application. Often it may\nsimply mean: the largest scalar product. That is, given a query <span class=\"math notranslate nohighlight\">\\(q\\in\\mathbb R^d\\)</span> we look for the <span class=\"math notranslate nohighlight\">\\(x^{(i)}\\)</span>\nwith largest <span class=\"math notranslate nohighlight\">\\(\\trsp{q}x^{(i)}\\)</span>. In that case, the problem is known as <em>maximum inner product search</em> (or MIPS).</p>", "a[href=\"#equation-vqobj\"]": "<div class=\"math notranslate nohighlight\" id=\"equation-vqobj\">\n\\[\\expec_{q\\sim\\mathcal D}\\left[\\sum_{i=1}^n(\\trsp q_jx^{(i)}_j-\\trsp q_j\\hat x^{(i)}_j)^2\\right]\\]</div>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,
                onShow(instance) {MathJax.typesetPromise([instance.popper]).then(() => {});},
            });
        };
    };
    console.log("tippy tips loaded!");
};
