function render({ model, el }) {

let container = document.createElement("div")
let containerElements = document.createElement("div")
let containerImages = document.createElement("div")

container.appendChild(containerElements)
container.appendChild(containerImages)

let elements = model.get("elements")
let settings = model.get("settings")
let wavefront_images = model.get("wavefront_images")

containerElements.className = 'svetlanna-elements-el-container'

containerImages.innerHTML = `
<div style="display: flex;justify-content: center;">
<img id="img" src="" style="width: auto; max-height: 12rem">
</div>
`

function insertElementSpecsHtml(el, i) {
    el.querySelector('#img').src = `data:image/png;base64,${wavefront_images[i]}`
}

function addPicker(i) {
    let elementFieldDiv = document.createElement("div")
    elementFieldDiv.style = "width:0;height:0;"
    elementFieldDiv.innerHTML = '<div class="svetlanna-field-picker clickable"></div>'

    elementFieldDiv.onclick = () => {
        insertElementSpecsHtml(containerImages, i+1)
    }
    
    containerElements.appendChild(elementFieldDiv)
}

addPicker(-1)

elements.forEach((element, i) => {
    let elementDiv = document.createElement("div")
    elementDiv.className = 'svetlanna-element'
    if (element.type == 'FreeSpace') {
        elementDiv.className = 'svetlanna-element svetlanna-element-free-space'
    }
    elementDiv.innerHTML = `<div class="svetlanna-element-index">(${element.index})</div>`
    containerElements.appendChild(elementDiv)
    addPicker(i)
});

el.appendChild(container);
}
export default { render };
