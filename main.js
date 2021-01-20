p5.disableFriendlyErrors = true;
const util = new p5(p => {
    p.setup = () => {
        p.createCanvas(0, 0);
    }
});
let pca;
let encoder;
let decoder;
let pcaSummary;
let encodings;
const cavnas = document.getElementById("canvas");
const ctx = cavnas.getContext("2d");
const render = document.getElementById("render");
const random = document.getElementById("random");
const sliders = document.getElementById("sliders");
const liveRender = document.getElementById("liveRender");
const imageUpload = document.getElementById("imageUpload");
const encode = document.getElementById("encode");
const save = document.getElementById("save");
const catName = document.getElementById("catName");
const average = document.getElementById("average");
let currCat = Array(256).fill(0);
let sliderList = [];
async function main() {
    encoder = await tf.loadLayersModel("encoder/model.json");
    decoder = await tf.loadLayersModel("decoder/model.json");
    const pcaJSON = await fetch("pca.json");
    pca = ML.PCA.load(await pcaJSON.json());
    pca.invert = function invert(dataset) {
        dataset = ML.Matrix.checkMatrix(dataset);

        var inverse = dataset.mmul(this.U.transpose());

        if (this.center) {
            if (this.scale) {
                inverse.mulRowVector(this.stdevs);
            }
            inverse.addRowVector(this.means);
        }
        return inverse;
    }
    pcaSummary = await fetch("pcaSummary.json");
    pcaSummary = await pcaSummary.json();
    encodings = await fetch("encodings.json");
    encodings = await encodings.json();
    sliders.innerHTML = "";
    for (let i = 0; i < pcaSummary.length; i++) {
        const label = document.createElement("label");
        label.innerHTML = `PC #${i + 1}`;
        if (i < 9) {
            label.innerHTML += "&nbsp;&nbsp;&nbsp;&nbsp;";
        } else if (i < 99) {
            label.innerHTML += "&nbsp;&nbsp;";
        }
        sliders.appendChild(label);
        const slider = document.createElement("input");
        slider.setAttribute("type", "range");
        slider.setAttribute("min", "0");
        slider.setAttribute("max", "100");
        slider.onchange = () => {
            const adjustedValue = util.map(+slider.value, 0, 100, -pcaSummary[i].stddiv, pcaSummary[i].stddiv, true);
            currCat[i] = adjustedValue;
            if (liveRender.checked) {
                renderCat();
            }
        }
        sliders.appendChild(slider);
        sliderList.push(slider);
        sliders.appendChild(document.createElement("br"));
    }
}
main();
const renderCat = () => {
    const image = decoder.predict(tf.tensor(pca.invert([currCat]).to2DArray())).arraySync()[0];
    for (let y = 0; y < 64; y++) {
        for (let x = 0; x < 64; x++) {
            ctx.fillStyle = `rgb(${image[x][y][0] * 255 }, ${image[x][y][1] * 255}, ${image[x][y][2] * 255})`;
            ctx.fillRect(y * 4, x * 4, 4, 4);
        }
    }
}
let firstInterval = setInterval(() => {
    if (decoder && pca) {
        renderCat();
        clearInterval(firstInterval);
    }
});
render.onclick = () => {
    if (decoder && pca) {
        render.innerHTML = "Rendering...";
        render.setAttribute("disabled", "true");
        setTimeout(() => {
            renderCat();
            render.innerHTML = "Render";
            render.removeAttribute("disabled");
        });
    }
};
random.onclick = () => {
    if (encodings && decoder && pca && pcaSummary) {
        const chosenVec = pca.predict([encodings[Math.floor(Math.random() * encodings.length)]]).to1DArray();
        currCat = chosenVec;
        sliderList.forEach((slider, i) => {
            slider.value = util.map(chosenVec[i], -pcaSummary[i].stddiv, pcaSummary[i].stddiv, 0, 100, true);
        })
        renderCat();
    }
}
encode.onclick = () => {
    if (imageUpload.files[0] && encoder) {
        const img = document.createElement("img");
        img.classList.add("obj");
        img.file = imageUpload.files[0];
        //img.width = 64;
        //img.height = 64;
        const reader = new FileReader();
        reader.onload = function(e) {
            img.src = e.target.result;
            setTimeout(() => {
                const tempCanvas = document.createElement("canvas");
                tempCanvas.width = 64;
                tempCanvas.height = 64;
                const tempCtx = tempCanvas.getContext("2d");
                tempCtx.drawImage(img, 0, 0, 64, 64);
                //const imageData = tempCtx.getImageData(0, 0, 64, 64).data.filter((x, i) => (i + 1) % 4 !== 0);
                const imageTensor = [];
                for (let y = 0; y < 64; y++) {
                    imageTensor[y] = [];
                    for (let x = 0; x < 64; x++) {
                        //const idx = (64 * y * x);
                        //imageTensor[y][x] = [imageData[idx] / 255, imageData[idx + 1] / 255, imageData[idx + 2] / 255]
                        imageTensor[y][x] = Array.from(tempCtx.getImageData(x, y, 1, 1).data.slice(0, 3)).map(x => x / 255);
                    }
                }
                const encoding = encoder.predict(tf.tensor([imageTensor])).arraySync()[0];
                const chosenVec = pca.predict([encoding]).to1DArray();
                currCat = chosenVec;
                sliderList.forEach((slider, i) => {
                    slider.value = util.map(chosenVec[i], -pcaSummary[i].stddiv, pcaSummary[i].stddiv, 0, 100, true);
                })
                renderCat();
            })
        };
        reader.readAsDataURL(imageUpload.files[0]);
    }
}
save.onclick = () => {
    canvas.toBlob(function(blob) {
        saveAs(blob, catName.value + ".png");
    });
}
average.onclick = () => {
    const chosenVec = Array(256).fill(0);
    currCat = chosenVec;
    sliderList.forEach((slider, i) => {
        slider.value = util.map(chosenVec[i], -pcaSummary[i].stddiv, pcaSummary[i].stddiv, 0, 100, true);
    })
    renderCat();
}