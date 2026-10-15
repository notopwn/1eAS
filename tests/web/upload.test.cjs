const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function page() {
  const elements = new Map();
  const allocations = [];
  const context = {drawImage() {}, clearRect() {}, fillRect() {},
    getImageData: () => ({data: new Uint8ClampedArray(224 * 224 * 4)})};
  function element() {
    let width = 0, height = 0;
    const e = {textContent: '', value: 'chosen', classList: {toggle() {}}, style: {},
      addEventListener() {}, setAttribute() {}, replaceChildren() {this.cleared = true;}, getContext: () => context};
    Object.defineProperties(e, {
      width: {get: () => width, set: v => {width = v; allocations.push([width, height]);}},
      height: {get: () => height, set: v => {height = v; allocations.push([width, height]);}}
    });
    return e;
  }
  class BrokenImage {
    set src(value) {queueMicrotask(() => this.onerror(new Error('decode failed')));}
  }
  const scope = vm.createContext({document: {
    getElementById: id => {if (!elements.has(id)) elements.set(id, element()); return elements.get(id);},
    createElement: element,
  }, Image: BrokenImage, Float32Array, Uint8ClampedArray, Number, Math, Map, Promise,
    console: {error() {}}, requestAnimationFrame: cb => cb(), window: {addEventListener() {}},
    URL: {createObjectURL: () => 'blob:bad', revokeObjectURL() {}},
    ort: {env: {wasm: {}}, Tensor: class {}}, setTimeout});
  const source = fs.readFileSync(path.join(__dirname, '../../docs/app.js'), 'utf8')
    .replace(/selectModel\("clean"\);\s*$/, '');
  vm.runInContext(source, scope);
  return {scope, elements, allocations};
}

test('48MP phone dimensions use bounded canvases and the expected model resize', () => {
  const p = page();
  const result = vm.runInContext('resizeShorterSide({naturalWidth: 8064, naturalHeight: 6048}, 256)', p.scope);
  assert.equal(result.width, 341);
  assert.equal(result.height, 256);
  assert.ok(p.allocations.every(([w, h]) => w <= 4096 && h <= 4096));
});

test('unusually thin panoramas fail with an actionable message', () => {
  const p = page();
  assert.throws(() => vm.runInContext('resizeShorterSide({naturalWidth: 20000, naturalHeight: 100}, 256)', p.scope), /Crop closer/);
});

test('unreadable uploads show an error and allow the same file to be selected again', async () => {
  const p = page();
  await p.elements.get('file').onchange({target: {files: [{name: 'photo.HEIC'}]}});
  assert.match(p.elements.get('predictionStatus').textContent, /JPEG or PNG/);
  assert.equal(p.elements.get('file').value, '');
  assert.equal(p.elements.get('preview').hidden, true);
});

test('inference failure is visible rather than leaving unchanged prediction bars', async () => {
  const p = page();
  await vm.runInContext(`model = {input: {resize:256,crop:224,mean:[0,0,0],std:[1,1,1],name:'image'}};
    session = {run: async () => {throw new Error('image memory exhausted');}};
    classify({width:512,height:512});`, p.scope);
  assert.match(p.elements.get('predictionStatus').textContent, /Prediction failed: image memory exhausted/);
  assert.equal(p.elements.get('bars').cleared, true);
});

test('a completed older inference cannot overwrite a newly selected image status', async () => {
  const p = page();
  const pending = vm.runInContext(`let finish;
    model = {input:{resize:256,crop:224,mean:[0,0,0],std:[1,1,1],name:'image'},output:{name:'logits'}};
    session = {run: () => new Promise(resolve => {finish=resolve;})};
    classify({width:512,height:512});`, p.scope);
  await new Promise(resolve => setImmediate(resolve));
  vm.runInContext(`lastSource=null; predictionStatus('Reading a new photo…');
    finish({logits:{data:[1,2,3,4,5]}});`, p.scope);
  await pending;
  assert.equal(p.elements.get('predictionStatus').textContent, 'Reading a new photo…');
});

test('leaving the camera panel stops its stream before uploads', () => {
  const p = page(); let stopped=false;
  vm.runInContext('cameraRunning=true;', p.scope);
  p.scope.document.getElementById('video').srcObject={getTracks:()=>[{stop(){stopped=true;}}]};
  vm.runInContext("showPanel('File');", p.scope);
  assert.ok(stopped);
  assert.equal(p.elements.get('video').srcObject, null);
  assert.equal(p.elements.get('camStart').disabled, false);
});
