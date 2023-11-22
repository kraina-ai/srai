const SELECTABILITY_PROPERTIES = [
    "user-select",
    "-webkit-user-select",
    "-ms-user-select",
    "-moz-user-select"
];

document$.subscribe(function () {
    makeButtonsCopySelectableOnly();
})

function makeButtonsCopySelectableOnly() {
    const buttonsToFix = document.querySelectorAll(".highlight button.md-clipboard");
    buttonsToFix.forEach((button) => {
        button.dataset.clipboardText = extractText(button.dataset.clipboardTarget);
    });
}

function extractText(selector) {
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
