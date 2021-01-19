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
    for (let i = 0; i < pcaSummary.length; i++) {
        const label = document.createElement("label");
        label.innerHTML = `PCA #${i + 1}`;
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