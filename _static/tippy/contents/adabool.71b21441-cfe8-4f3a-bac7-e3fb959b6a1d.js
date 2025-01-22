selector_to_html = {"a[href=\"#adaboost-cannot-fit-every-boolean-function\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">AdaBoost cannot fit every Boolean function<a class=\"headerlink\" href=\"#adaboost-cannot-fit-every-boolean-function\" title=\"Link to this heading\">#</a></h1><p>AdaBoost is a type of additive model that combines weak classifiers in a weighted sum.\nWe denote by <span class=\"math notranslate nohighlight\">\\(\\mathcal W\\)</span> the set of weak classifiers. Often the set <span class=\"math notranslate nohighlight\">\\(\\mathcal W\\)</span> consists\nof <em>decision stumps</em>, ie, of classifiers that are defined by a dimension <span class=\"math notranslate nohighlight\">\\(i\\in\\{1,\\ldots, n\\}\\)</span>,\na threshold <span class=\"math notranslate nohighlight\">\\(t\\in\\mathbb R\\)</span> and a sign <span class=\"math notranslate nohighlight\">\\(\\sigma\\in\\{-1,1\\}\\)</span>:</p>", "a[href=\"#plusminusloss\"]": "<div class=\"math notranslate nohighlight\" id=\"plusminusloss\">\n<span id=\"equation-plusminusloss\"></span>\\[0=\\sum_{i=1}^k\\sum_{(x,y)\\in S}\\alpha_iyh_i(x) = \\sum_{(x,1)\\in S}\\sum_{i=1}^k\\alpha_ih_i(x)-\\sum_{(x,-1)\\in S}\\sum_{i=1}^k\\alpha_ih_i(x) \n\\rlap{\\qquad(A)}\\]</div>"}
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
