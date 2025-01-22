selector_to_html = {"a[href=\"#short-stochastic-digression\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">8.2. </span>Short stochastic digression<a class=\"headerlink\" href=\"#short-stochastic-digression\" title=\"Link to this heading\">#</a></h2><p>To cope with a collective of classifiers with stochastic interdependencies\nwe need a stochastic tool, a sort of  one-sided\nChebyshev\u2019s inequality.</p><p>For comparison, let\u2019s recall Chebyshev\u2019s inequality.\nFor this let <span class=\"math notranslate nohighlight\">\\(X\\)</span> be a random variable, and recall the definition\nof the <em>variance</em> of a random variable:</p>", "a[href=\"#wisdom-of-the-crowd\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">8.1. </span>Wisdom of the crowd<a class=\"headerlink\" href=\"#wisdom-of-the-crowd\" title=\"Link to this heading\">#</a></h2><p>An easy way to increase the performance in a classification task is\nto train several classifiers and then let them decide by majority voting.</p><p>To gain a first insight, consider a binary classification task and\nassume that we have access to <span class=\"math notranslate nohighlight\">\\(T\\)</span> classifiers\n<span class=\"math notranslate nohighlight\">\\(h_1,\\ldots, h_T\\)</span> that each have a probability of <span class=\"math notranslate nohighlight\">\\(p&gt;\\tfrac{1}{2}\\)</span>\nto classify a randomly drawn data point correctly (we assume here that\nthe class <span class=\"math notranslate nohighlight\">\\(y\\)</span> is completely determined by <span class=\"math notranslate nohighlight\">\\(x\\)</span>). Assume, furthermore,\nthat the classifiers are stochastically independent. (Clearly,\nthis is an unrealistic assumption.) Then the probability that\nthe majority vote decides wrongly is</p>", "a[href=\"#ensemble-learning\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">8. </span>Ensemble learning<a class=\"headerlink\" href=\"#ensemble-learning\" title=\"Link to this heading\">#</a></h1><p>In <em>Who wants to be a millionaire?</em> contestants have the option to ask the audience\nfor help with a quiz question. Often this successful: the majority vote indicates the right answer.\nThe interesting feature\nhere is that obviously the audience of game shows does not usually consist of experts.\nOn the contrary, the typical audience member  is arguably more ignorant than the contestants,\nwho have already proved their merit by clearing the pre-selection process. Still,\ncollectively, the audience is relatively strong.\nThis phenomenon is called <em>wisdom of the crowd</em>.<label class=\"margin-toggle marginnote-label\" for=\"marginnote-role-1\"></label><input class=\"margin-toggle\" id=\"marginnote-role-1\" name=\"marginnote-role-1\" type=\"checkbox\"/><span class=\"marginnote\"> There seems to have been written a lot about the supposed\nwisdom of the crowd, even a whole book. Sometimes, though, the crowd is dead-wrong:\nThe 80s, for example, were terrible and there\u2019s no reason to celebrate them.</span></p><p><em>Ensemble learning</em> combines several weak\npredictors to a strong predictor. Broadly, there are two ways to do that:</p>"}
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
