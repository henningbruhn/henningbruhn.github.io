selector_to_html = {"a[href=\"#example-of-correlated-classifiers\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Example of correlated classifiers<a class=\"headerlink\" href=\"#example-of-correlated-classifiers\" title=\"Link to this heading\">#</a></h1><p>Let\u2019s look at an example. We fix the domain to <span class=\"math notranslate nohighlight\">\\(\\mathcal X=[0,2]^3\\)</span>,\nand the distribution <span class=\"math notranslate nohighlight\">\\(\\mathcal D\\)</span> so that the marginal probability <span class=\"math notranslate nohighlight\">\\(\\proba[x]\\)</span>\nis uniform on <span class=\"math notranslate nohighlight\">\\(\\mathcal X\\)</span>, and so that the data completely determines the class.\nWe set</p>"}
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
