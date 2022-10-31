// document.body.onload = call_fun;
// function inputExpandCollapse() {
//   let copy_button = document.querySelectorAll(".expandClass");
//   for (let i = 0; i < copy_button.length; i++) {
//     copy_button[i].addEventListener("click", function () {
//       this.classList.toggle("active");
//       var content = document.getElementsByClassName("highlight-ipynb");
//       var elem2 = document.getElementsByClassName("zeroclipboard-container");
//       if (content[i].style.display === "block") {
//         content[i].style.display = "none";
//         elem2[i].style.setProperty("position", "relative", "important");
//       } else {
//         content[i].style.display = "block";
//         elem2[i].style.setProperty("position", "absolute", "important");
//       }
//     });
//   }
// }
// function insertNew() {
//   // create new elements
//   var button = document.createElement("button");
//   button.innerHTML = "+";
//   button.setAttribute("class", "expandClass");
//   // append another one
//   const elem2 = document.getElementsByClassName("zeroclipboard-container");
//   for (let i = 0; i < elem2.length; i++) {
//     elem2[i].appendChild(button.cloneNode(true));
//   }
//   // insert new id
//   var code_div = document.getElementsByClassName("highlight-ipynb");
//   for (let i = 0; i < code_div.length; i++) {
//     code_div[i].setAttribute("id", "code-cell" + "-" + i.toString());
//   }
//   // select clipboard-copy element
//   var clipboard_button = document.querySelectorAll("clipboard-copy");
//   for (let i = 0; i < clipboard_button.length; i++) {
//     clipboard_button[i].setAttribute("for", "code-cell" + "-" + i.toString());
//   }
// }
// function call_fun() {
//   insertNew();
//   inputExpandCollapse();
// }
