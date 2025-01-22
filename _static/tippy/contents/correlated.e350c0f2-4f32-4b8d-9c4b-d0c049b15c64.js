selector_to_html = {"a[href=\"#ensembleexfig\"]": "<figure class=\"align-default\" id=\"ensembleexfig\">\n<a class=\"reference internal image-reference\" href=\"../_images/correlated.png\"><img alt=\"../_images/correlated.png\" src=\"../_images/correlated.png\" style=\"height: 6cm;\"/>\n</a>\n<figcaption>\n<p><span class=\"caption-text\">Class 1 in grey. Three digit binary strings encode whether <span class=\"math notranslate nohighlight\">\\(h_1,h_2,h_3\\)</span> are correct\non the corresponding cube.</span><a class=\"headerlink\" href=\"#ensembleexfig\" title=\"Link to this image\">#</a></p>\n</figcaption>\n</figure>", "a[href=\"#example-of-correlated-classifiers\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Example of correlated classifiers<a class=\"headerlink\" href=\"#example-of-correlated-classifiers\" title=\"Link to this heading\">#</a></h1><p>Let\u2019s look at an example. We fix the domain to <span class=\"math notranslate nohighlight\">\\(\\mathcal X=[0,2]^3\\)</span>,\nand the distribution <span class=\"math notranslate nohighlight\">\\(\\mathcal D\\)</span> so that the marginal probability <span class=\"math notranslate nohighlight\">\\(\\proba[x]\\)</span>\nis uniform on <span class=\"math notranslate nohighlight\">\\(\\mathcal X\\)</span>, and so that the data completely determines the class.\nWe set</p>"}
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
