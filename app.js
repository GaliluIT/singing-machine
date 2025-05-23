import { FilesetResolver, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.12.0/dist/tf.esm.js';

// Wait for DOM to be fully loaded before grabbing elements
window.addEventListener('DOMContentLoaded', () => {
  // Grab HTML elements after they exist
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  console.log('🛠️ DOM loaded, video element:', video);

  let faceLandmarker = null;

  // Request webcam access and start video
  async function setupCamera() {
    console.log('🔄 Requesting webcam access...');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();  // ensures the play promise resolves
    console.log('✅ Webcam stream started');
  }

  // Load the MediaPipe Face Landmarker model
  async function loadModel() {
    console.log('🔄 Loading face landmarker model...');
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      outputFaceBlendshapes: true,
      runningMode: 'VIDEO',
      numFaces: 1
    });
    console.log('✅ Model loaded');
  }

  // Draw loop: detect landmarks and render
  async function render() {
    if (!faceLandmarker) return;
    const results = await faceLandmarker.detectForVideo(video, Date.now());
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];
      for (const point of landmarks) {
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 2, 0, 2 * Math.PI);
        ctx.fillStyle = 'cyan';
        ctx.fill();
      }
    }
    requestAnimationFrame(render);
  }

  // Initialize sequence
  (async () => {
    try {
      await setupCamera();
      await loadModel();
      render();
    } catch (err) {
      console.error('❌ Initialization error:', err);
      alert('Error accessing webcam or loading model. Check console for details.');
    }
  })();
});
