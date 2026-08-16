const input = document.getElementById("image");
if (input) {
  input.addEventListener("change", () => {
    const label = input.parentElement.querySelector("span");
    if (label && input.files[0]) {
      label.textContent = input.files[0].name;
    }
  });
}
