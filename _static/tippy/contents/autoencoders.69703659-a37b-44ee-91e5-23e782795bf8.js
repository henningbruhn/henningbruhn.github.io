selector_to_html = {"a[href=\"#autoencoders\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">7. </span>Autoencoders<a class=\"headerlink\" href=\"#autoencoders\" title=\"Link to this heading\">#</a></h1><p>Labelled training data is often scarce. Labelling is typically costly and time consuming. In many situations, however,\nthere a large number of unlabelled data is available. <em>Autoencoders</em> are a way to leverage unlabelled data\nin task that in principle would require labelled data. So what\u2019s an autoencoder? An autoencoder\nconsists of two parts: an <em>encoder</em> neural network <span class=\"math notranslate nohighlight\">\\(e:\\mathbb R^n\\to\\mathbb R^k\\)</span> and\na <em>decoder</em> neural network <span class=\"math notranslate nohighlight\">\\(d:\\mathbb R^k\\to\\mathbb R^n\\)</span>. Here, <span class=\"math notranslate nohighlight\">\\(n\\)</span> is the input dimension, while\n<span class=\"math notranslate nohighlight\">\\(k\\)</span> is the dimension of the <em>latent space</em>, ie, of the <em>latent representation</em> <span class=\"math notranslate nohighlight\">\\(z=e(x)\\)</span>.\nThe idea of an autoencoder is that it learns to replicate the input. That is, that on input <span class=\"math notranslate nohighlight\">\\(x\\)</span> it computes</p>", "a[href=\"#autoencoderfig\"]": "<figure class=\"align-default\" id=\"autoencoderfig\">\n<a class=\"reference internal image-reference\" href=\"../_images/autoencoder.png\"><img alt=\"../_images/autoencoder.png\" src=\"../_images/autoencoder.png\" style=\"height: 6cm;\"/>\n</a>\n<figcaption>\n<p><span class=\"caption-number\">Fig. 7.1 </span><span class=\"caption-text\">An autoencoder.</span><a class=\"headerlink\" href=\"#autoencoderfig\" title=\"Link to this image\">#</a></p>\n</figcaption>\n</figure>"}
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
