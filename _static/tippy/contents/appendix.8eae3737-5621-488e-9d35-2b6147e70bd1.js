selector_to_html = {"a[href=\"#very-basic-probability-theory\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">11.1. </span>Very basic probability theory<a class=\"headerlink\" href=\"#very-basic-probability-theory\" title=\"Link to this heading\">#</a></h2><p>A finite probability space  consists of\nof a finite <em>sample space</em> <span class=\"math notranslate nohighlight\">\\(\\Omega\\)</span> and a <em>probability measure</em> <span class=\"math notranslate nohighlight\">\\(\\proba:2^\\Omega\\to [0,1]\\)</span>\nthat satisfies the following properties:</p>", "a[href=\"#appendix\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">11. </span>Appendix<a class=\"headerlink\" href=\"#appendix\" title=\"Link to this heading\">#</a></h1><h2><span class=\"section-number\">11.1. </span>Very basic probability theory<a class=\"headerlink\" href=\"#very-basic-probability-theory\" title=\"Link to this heading\">#</a></h2><p>A finite probability space  consists of\nof a finite <em>sample space</em> <span class=\"math notranslate nohighlight\">\\(\\Omega\\)</span> and a <em>probability measure</em> <span class=\"math notranslate nohighlight\">\\(\\proba:2^\\Omega\\to [0,1]\\)</span>\nthat satisfies the following properties:</p>", "a[href=\"#stdnormalfig\"]": "<figure class=\"align-default\" id=\"stdnormalfig\">\n<a class=\"reference internal image-reference\" href=\"../_images/stdnormalfill.png\"><img alt=\"../_images/stdnormalfill.png\" src=\"../_images/stdnormalfill.png\" style=\"width: 12cm;\"/>\n</a>\n<figcaption>\n<p><span class=\"caption-number\">Fig. 11.1 </span><span class=\"caption-text\">The univariate standard normal distribution. Grey area: The probability that the outcome is in <span class=\"math notranslate nohighlight\">\\([1.2,1.5]\\)</span>.</span><a class=\"headerlink\" href=\"#stdnormalfig\" title=\"Link to this image\">#</a></p>\n</figcaption>\n</figure>"}
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
