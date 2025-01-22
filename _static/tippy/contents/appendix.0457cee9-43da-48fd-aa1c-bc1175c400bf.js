selector_to_html = {"a[href=\"#appendix\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">9. </span>Appendix<a class=\"headerlink\" href=\"#appendix\" title=\"Link to this heading\">#</a></h1><h2><span class=\"section-number\">9.1. </span>Very basic probability theory<a class=\"headerlink\" href=\"#very-basic-probability-theory\" title=\"Link to this heading\">#</a></h2><p>A finite probability space  consists of\nof a finite <em>sample space</em> <span class=\"math notranslate nohighlight\">\\(\\Omega\\)</span> and a <em>probability measure</em> <span class=\"math notranslate nohighlight\">\\(\\proba:2^\\Omega\\to [0,1]\\)</span>\nthat satisfies the following properties:</p>", "a[href=\"#very-basic-probability-theory\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">9.1. </span>Very basic probability theory<a class=\"headerlink\" href=\"#very-basic-probability-theory\" title=\"Link to this heading\">#</a></h2><p>A finite probability space  consists of\nof a finite <em>sample space</em> <span class=\"math notranslate nohighlight\">\\(\\Omega\\)</span> and a <em>probability measure</em> <span class=\"math notranslate nohighlight\">\\(\\proba:2^\\Omega\\to [0,1]\\)</span>\nthat satisfies the following properties:</p>"}
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
