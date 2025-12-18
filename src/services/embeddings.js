const sharp = require("sharp");
const tf = require("@tensorflow/tfjs");
const mobilenet = require("@tensorflow-models/mobilenet");

let model;

async function ensureMobilenet() {
  if (model) return model;
  await tf.setBackend("cpu");
  await tf.ready();
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  return model;
}

async function embedImageFile(filepath) {
  if (!model) await ensureMobilenet();

  const { data, info } = await sharp(filepath)
    .removeAlpha()
    .resize(224, 224, { fit: "cover" })
    .raw()
    .toBuffer({ resolveWithObject: true });

  return tf.tidy(() => {
    const uint8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const tensor = tf.tensor3d(uint8, [info.height, info.width, 3], "int32");
    const batched = tensor.expandDims(0);
    const embeddings = model.infer(batched, "conv_preds");
    const arr = embeddings.dataSync();
    return Array.from(arr);
  });
}

function cosineSimilarity(a, b) {
  let dot = 0,
    na = 0,
    nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const x = a[i] || 0,
      y = b[i] || 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

module.exports = {
  ensureMobilenet,
  embedImageFile,
  cosineSimilarity,
};
