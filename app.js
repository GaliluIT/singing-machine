{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 const video = document.getElementById('webcam');\
const canvas = document.getElementById('output');\
const ctx = canvas.getContext('2d');\
\
let faceLandmarker;\
let running = false;\
\
// Load the face landmark model\
async function loadModel() \{\
  const vision = await FilesetResolver.forVisionTasks(\
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'\
  );\
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, \{\
    baseOptions: \{\
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',\
      delegate: 'GPU'\
    \},\
    outputFaceBlendshapes: true,\
    runningMode: 'VIDEO',\
    numFaces: 1\
  \});\
\}\
\
// Start webcam\
async function setupCamera() \{\
  const stream = await navigator.mediaDevices.getUserMedia(\{ video: true \});\
  video.srcObject = stream;\
  await new Promise(resolve => video.onloadedmetadata = resolve);\
\}\
\
// Analyze video frame-by-frame\
async function render() \{\
  if (!faceLandmarker) return;\
\
  const results = await faceLandmarker.detectForVideo(video, Date.now());\
\
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);\
\
  if (results.faceLandmarks.length > 0) \{\
    const landmarks = results.faceLandmarks[0];\
    landmarks.forEach(point => \{\
      ctx.beginPath();\
      ctx.arc(point.x * canvas.width, point.y * canvas.height, 2, 0, 2 * Math.PI);\
      ctx.fillStyle = 'cyan';\
      ctx.fill();\
    \});\
  \}\
\
  requestAnimationFrame(render);\
\}\
\
(async () => \{\
  await setupCamera();\
  await loadModel();\
  render();\
\})();\
}