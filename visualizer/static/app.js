import * as THREE from './vendor/three.module.js';

const $ = (id) => document.getElementById(id);
const canvas = $('sphereCanvas');
let busy = false, loaded = false, embedded = false, dragging = false, lastVector;
let renderer, scene, camera, sphere;
const confirmed = new THREE.Quaternion();
const textureLoader = new THREE.TextureLoader();

function status(message) { $('statusText').textContent = message; }
function refreshButtons() {
  for (const el of document.querySelectorAll('button, input, select')) el.disabled = busy;
  $('embedBtn').disabled = busy || !loaded;
  $('confirmRotationBtn').disabled = busy || !embedded;
  $('resetRotationBtn').disabled = busy || !loaded;
}
async function request(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
async function run(message, task) {
  if (busy) return;
  busy = true; refreshButtons(); status(message);
  try { await task(); }
  catch (error) { status(error.message); console.error(error); }
  finally { busy = false; refreshButtons(); }
}
function quaternionReadout() {
  $('quatReadout').textContent = sphere.quaternion.toArray().map(x => x.toFixed(3)).join(', ');
}
function resetOutputs() {
  for (const id of ['cleanBer', 'cleanAcc', 'rotBer', 'rotAcc']) $(id).textContent = '-';
  for (const id of ['watermarkedPreview', 'rotatedPreview']) $(id).removeAttribute('src');
  for (const id of ['watermarkedLink', 'rotatedLink', 'jsonLink']) $(id).href = '#';
  embedded = false;
}
async function texture(url) {
  const map = await textureLoader.loadAsync(url);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = THREE.RepeatWrapping;
  map.minFilter = THREE.LinearFilter;
  map.generateMipmaps = false;
  sphere.material.map?.dispose();
  sphere.material.map = map;
  sphere.material.color.set(0xffffff);
  sphere.material.needsUpdate = true;
}
async function showLoaded(data) {
  resetOutputs();
  $('originalPreview').src = data.original_url;
  await texture(data.original_url);
  sphere.quaternion.identity(); confirmed.identity(); quaternionReadout();
  loaded = true;
  $('sessionText').textContent = data.session;
  status(data.image_name);
}
async function listImages() {
  const data = await request('/api/images');
  $('imageSelect').replaceChildren(...data.images.map(item => new Option(item.name, item.path)));
  status(`${data.images.length} panoramas`);
}
function trackball(event) {
  const rect = canvas.getBoundingClientRect();
  const size = Math.min(rect.width, rect.height) * 0.43;
  const x = (event.clientX - rect.left - rect.width / 2) / size;
  const y = -(event.clientY - rect.top - rect.height / 2) / size;
  return new THREE.Vector3(x, y, Math.sqrt(Math.max(0, 1-x*x-y*y))).normalize().applyQuaternion(camera.quaternion);
}
function initSphere() {
  renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0xf0f3f5);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 20);
  camera.up.set(0, 0, 1);
  camera.position.set(0, -3.5, 0);
  camera.lookAt(0, 0, 0);
  // Remap Three's Y-up sphere to z-up ERP: longitude starts at +x.
  const geometry = new THREE.SphereGeometry(1, 128, 64);
  const position = geometry.attributes.position;
  const uv = geometry.attributes.uv;
  for (let i = 0; i < position.count; i++) {
    const theta = (1 - uv.getY(i)) * Math.PI;
    const phi = uv.getX(i) * 2 * Math.PI;
    position.setXYZ(i, Math.sin(theta)*Math.cos(phi), Math.sin(theta)*Math.sin(phi), Math.cos(theta));
  }
  geometry.computeVertexNormals();
  sphere = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: 0x96b2b0, side: THREE.DoubleSide}));
  scene.add(sphere);
  function render() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (w && h) {
      renderer.setSize(w, h, false);
      camera.aspect = w/h;
      camera.position.y = -3.5 * Math.max(1, 1/camera.aspect);
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    }
    requestAnimationFrame(render);
  }
  render();
  canvas.addEventListener('pointerdown', e => {
    if (busy || !loaded) return;
    dragging = true; lastVector = trackball(e); canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging || busy) return;
    const next = trackball(e);
    sphere.quaternion.premultiply(new THREE.Quaternion().setFromUnitVectors(lastVector, next)).normalize();
    lastVector = next; quaternionReadout();
    if (sphere.quaternion.angleTo(confirmed) > 1e-5) status('Rotation pending');
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(name, () => { dragging = false; });
  }
}

async function boot() {
  initSphere(); refreshButtons();
  const config = await request('/api/config');
  $('runtimeInfo').textContent = `${config.device} | ${config.height} x ${config.width} | ${config.model_info.latent_dim} bits`;
  await listImages();
  $('refreshImagesBtn').onclick = () => run('Loading...', listImages);
  $('loadImageBtn').onclick = () => run('Loading panorama...', async () => {
    if (!$('imageSelect').value) throw new Error('Select or upload an ERP image');
    await showLoaded(await request('/api/load_path', {path: $('imageSelect').value}));
  });
  $('uploadInput').onchange = () => run('Uploading...', async () => {
    const file = $('uploadInput').files[0];
    if (!file) return;
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    await showLoaded(await request('/api/load_upload', {data_url: data, name: file.name}));
    $('uploadInput').value = '';
  });
  $('embedBtn').onclick = () => run('Embedding watermark...', async () => {
    const result = await request('/api/embed', {seed: Number($('seedInput').value)});
    resetOutputs();
    $('watermarkedPreview').src = result.watermarked_url;
    $('watermarkedLink').href = result.watermarked_url;
    $('cleanBer').textContent = result.clean_ber.toFixed(4);
    $('cleanAcc').textContent = (result.clean_accuracy*100).toFixed(2)+'%';
    await texture(result.watermarked_url);
    sphere.quaternion.identity(); confirmed.identity(); quaternionReadout();
    embedded = true; status('Watermark embedded');
  });
  $('resetRotationBtn').onclick = () => {
    sphere.quaternion.identity(); quaternionReadout(); status('Rotation reset');
  };
  $('confirmRotationBtn').onclick = () => run('Rotating and extracting...', async () => {
    const q = sphere.quaternion.clone();
    const result = await request('/api/rotate_extract', {rotation_quaternion: q.toArray()});
    $('rotatedPreview').src = result.rotated_url;
    $('rotatedLink').href = result.rotated_url;
    $('jsonLink').href = result.extraction_json;
    $('rotBer').textContent = result.ber.toFixed(4);
    $('rotAcc').textContent = (result.accuracy*100).toFixed(2)+'%';
    confirmed.copy(q);
    // Keep source texture and absolute rotation so repeated confirms agree.
    status(`Saved | ${result.errors}/${result.num_bits} bit errors`);
  });
}
boot().catch(error => { status(error.message); console.error(error); });
