document.addEventListener('DOMContentLoaded', (event) => {
    document.querySelectorAll('pre code').forEach((block) => {
        var button = document.createElement('button');
        button.className = 'copy-button';
        button.type = 'button';
        button.innerText = 'Copy';
        button.addEventListener('click', function () {
            navigator.clipboard.writeText(block.innerText).then(function () {
                /* clipboard successfully set */
                button.innerText = 'Copied!';
                setTimeout(function () {
                    button.innerText = 'Copy';
                }, 2000);
            }, function () {
                /* clipboard write failed */
                button.innerText = 'Failed to copy';
            });
        });
        block.parentNode.insertBefore(button, block);
    });
});