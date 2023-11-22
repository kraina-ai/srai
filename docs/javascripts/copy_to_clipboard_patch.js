const SELECTABILITY_PROPERTIES = [
    "user-select",
    "-webkit-user-select",
    "-ms-user-select",
    "-moz-user-select"
];

document$.subscribe(function () {
    fixCopyOnlyUserSelectable();
})

function fixCopyOnlyUserSelectable() {
    const buttonsToFix = document.querySelectorAll(".highlight button.md-clipboard");
    buttonsToFix.forEach((button) => {
        const content = extractUserSelectable(button.dataset.clipboardTarget);
        button.dataset.clipboardText = content;
    });
}

function extractUserSelectable(selector) {
    const element = document.querySelector(selector);
    return Array.from(element.childNodes)
        .filter(child => includeInOutput(child))
        .map(child => child.textContent)
        .join("")
        .trimEnd();
}

function includeInOutput(element) {
    if (element instanceof Element) {
        return isSelectable(element);
    }
    return true;
}

function isSelectable(element) {
    const childStyle = window.getComputedStyle(element);
    return !SELECTABILITY_PROPERTIES.some((prop) => childStyle.getPropertyValue(prop) == "none");
}
