// Do NOT import tfjs here
import { FilesetResolver, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';
console.log('TensorFlow.js version:', tf?.version?.tfjs);

// ...use tf.tensor2d, tf.sequential, etc, as usual

window.addEventListener('DOMContentLoaded', () => {
  // --- DOM refs ---
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  const userIDInput = document.getElementById('userID');
  const languageSelector = document.getElementById('language');
  const intendedVowelSelector = document.getElementById('intendedVowel');
  const addExampleBtn = document.getElementById('addExample');
  const trainModelBtn = document.getElementById('trainModel');
  const predictBtn = document.getElementById('predict');
  const saveModelBtn = document.getElementById('saveModel');
  const loadModelBtn = document.getElementById('loadModel');
  const exportModelBtn = document.getElementById('exportModel');
  const listModelsBtn = document.getElementById('listModels');
  const importInput = document.getElementById('importModel');
  const enableSoundCheckbox = document.getElementById('enableSound');
  const predictionSpan = document.getElementById('prediction');
  const rawDataOutput = document.getElementById('rawDataOutput');

  let faceLandmarker = null;
  let currentFeatures = [];
  const modelsStore = {};
  let audioContext = null, oscillator = null;
  const vowelFrequencies = { A:440, E:660, I:880, O:330, U:220 };
  let lastVowel = '';

  // --- Utility functions ---
  function getUserKey() {
    const user = userIDInput.value.trim() || 'default';
    const lang = languageSelector.value;
    return `${user}_${lang}`;
  }

  function playVowelSound(vowel) {
    if (!enableSoundCheckbox.checked || vowel === lastVowel) return;
    lastVowel = vowel;
    if (!audioContext) audioContext = new (window.AudioContext||window.webkitAudioContext)();
    if (oscillator) oscillator.stop();
    oscillator = audioContext.createOscillator();
    oscillator.type = 'sine';
    oscillator.frequency.value = vowelFrequencies[vowel] || 440;
    oscillator.connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.5);
  }

  function updateRawDataDisplay() {
    rawDataOutput.textContent = `Current Features:\n` +
      `Width: ${currentFeatures[0]?.toFixed(2)}\n` +
      `Height: ${currentFeatures[1]?.toFixed(2)}\n` +
      `Ratio: ${currentFeatures[2]?.toFixed(2)}`;
  }

  // --- Classifier event handlers ---
  addExampleBtn.addEventListener('click', () => {
    const key = getUserKey();
    const label = intendedVowelSelector.value;
    if (!modelsStore[key]) modelsStore[key] = { examples:[], labels:[], model:null };
    if (currentFeatures.length === 3) {
      modelsStore[key].examples.push(currentFeatures);
      modelsStore[key].labels.push({A:0,E:1,I:2,O:3,U:4}[label]);
      console.log(`Added example ${label} for ${key}`, currentFeatures);
    }
  });

  trainModelBtn.addEventListener('click', async () => {
    const key = getUserKey();
    const data = modelsStore[key];
    if(!data?.examples.length) return;
    const xs = tf.tensor2d(data.examples);
    const ys = tf.oneHot(tf.tensor1d(data.labels,'int32'),5);
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape:[3],units:16,activation:'relu'}));
    model.add(tf.layers.dense({units:16,activation:'relu'}));
    model.add(tf.layers.dense({units:5,activation:'softmax'}));
    model.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
    await model.fit(xs,ys,{epochs:30,shuffle:true});
    data.model = model;
    await model.save(`localstorage://${key}`);
    console.log(`Model trained & saved for ${key}`);
  });

  predictBtn.addEventListener('click', async () => {
    const key = getUserKey();
    const data = modelsStore[key];
    if(data?.model && currentFeatures.length === 3){
      const pred = data.model.predict(tf.tensor2d([currentFeatures]));
      const idx = (await pred.argMax(1).data())[0];
      const vowel = ['A','E','I','O','U'][idx];
      predictionSpan.innerText = `Predicted: ${vowel}`;
      playVowelSound(vowel);
    } else {
      predictionSpan.innerText = "No model or features!";
    }
  });

  saveModelBtn.addEventListener('click', async () => {
    const key = getUserKey();
    const data = modelsStore[key];
    if(data?.model){
      await data.model.save(`localstorage://${key}`);
      alert('Model saved!');
    }
  });

  loadModelBtn.addEventListener('click', async () => {
    const key = getUserKey();
    try {
      const m = await tf.loadLayersModel(`localstorage://${key}`);
      modelsStore[key] = modelsStore[key]||{examples:[],labels:[],model:null};
      modelsStore[key].model = m;
      console.log(`Loaded model for ${key}`);
    } catch {
      console.error(`No model found for ${key}`);
      alert(`No model found for ${key}`);
    }
  });

  exportModelBtn.addEventListener('click', async ()=>{
    const key = getUserKey(); 
    const m = modelsStore[key]?.model;
    if(!m) return alert(`No model to export for ${key}`);
    const handler = {
      async save(artifacts){
        const zip = new JSZip();
        zip.file('model.json',JSON.stringify(artifacts.modelTopology));
        zip.file('weights.bin',new Uint8Array(artifacts.weightData));
        const blob = await zip.generateAsync({type:'blob'});
        const link = document.createElement('a'); link.href=URL.createObjectURL(blob);
        link.download = `${key}_model.zip`; link.click();
        return {modelArtifactsInfo:{dateSaved:new Date(),modelTopologyType:'JSON',weightDataBytes:artifacts.weightData.byteLength}};
      }
    };
    await m.save(handler);
  });

  listModelsBtn.addEventListener('click', ()=>{
    const keys = Object.keys(localStorage).filter(k=>k.startsWith('tensorflowjs_models/'));
    alert('Saved models:\n'+keys.map(k=>k.replace('tensorflowjs_models/','')).join('\n'));
  });

  importInput.addEventListener('change', async e => {
    const file = e.target.files[0]; if(!file) return;
    const zip = await JSZip.loadAsync(file);
    const key = getUserKey();
    const modelJson = await zip.file('model.json').async('string');
    const weightData = await zip.file('weights.bin').async('arraybuffer');
    const handler = {load:async()=>({modelTopology:JSON.parse(modelJson),weightData,weightSpecs:[]})};
    const m = await tf.loadLayersModel(handler);
    modelsStore[key] = modelsStore[key]||{examples:[],labels:[],model:null};
    modelsStore[key].model = m;
    predictionSpan.innerText = `✅ Imported model for ${key}`;
  });

  // --- Webcam + MediaPipe setup ---
  async function setupCamera(){
    const stream = await navigator.mediaDevices.getUserMedia({video:true});
    video.srcObject = stream; await video.play();
  }

  async function loadFaceLandmarker(){
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision,{
      baseOptions:{
        modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate:'GPU'
      },
      outputFaceBlendshapes:true,
      runningMode:'VIDEO',
      numFaces:1
    });
  }

  async function predictWebcam() {
    const res = await faceLandmarker.detectForVideo(video, Date.now());
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (res.faceLandmarks.length) {
      const lm = res.faceLandmarks[0];
      lm.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 2, 0, 2 * Math.PI);
        ctx.fillStyle = 'cyan';
        ctx.fill();
      });
      const top = lm[13], bottom = lm[14], left = lm[61], right = lm[291];
      const h = Math.hypot(bottom.x - top.x, bottom.y - top.y) * canvas.height;
      const w = Math.hypot(right.x - left.x, right.y - left.y) * canvas.width;
      const ratio = h / w;
      currentFeatures = [w, h, ratio];
      updateRawDataDisplay();
      // Don't predict here!
    }
    requestAnimationFrame(predictWebcam);
  }
  
  // --- Init sequence ---
  (async()=>{
    try {
      await setupCamera();
      await loadFaceLandmarker();
      predictWebcam();
    } catch(e) {
      console.error(e);
      alert('Please allow webcam access and reload');
    }
  })();
});

