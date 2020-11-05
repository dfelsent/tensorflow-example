const tf = require('@tensorflow/tfjs-node');

const MODEL_PATH = 'file://./model.json'

async function makePrediction() {
    console.log('about to load model...');
    const myModel = await tf.loadLayersModel(MODEL_PATH);
    console.log('loaded model!');
    // 80 users, with 6 features each (all features have example value of 1)
    myModel.predict(tf.ones([80, 6])).flatten().print();
    // 1 user with 6 features of varying values
    // myModel.predict(tf.tensor2d([1, 0, 3, 4, 5, 6], [1, 6])).flatten().print();
}

makePrediction();