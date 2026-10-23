// Exercise the shipped export/render handlers without a browser or dependencies.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function workspace() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      classList: {add(){}, remove(){}, toggle(){}}, addEventListener(){}, click(){},
      innerHTML: '', textContent: '', disabled: false,
    });
    return elements.get(id);
  };
  let exported;
  const context = vm.createContext({
    document: {getElementById: element, querySelectorAll: () => [], addEventListener(){},
      createElement: () => ({click(){}})},
    window: {addEventListener(){}}, setTimeout: () => 1, clearTimeout(){},
    fetch: () => new Promise(() => {}),
    Blob: class {constructor(parts) {this.parts = parts;}},
    URL: {createObjectURL(blob) {exported = JSON.parse(blob.parts.join('')); return 'blob:fixture';}, revokeObjectURL(){}},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../studio/dist/app.js'), 'utf8'), context);
  const native = {
    version: 'source-measurements-1.0', sample_rate_hz: 48000, near_full_scale_threshold: .999,
    channels: [{channel: 1, rms_dbfs: null, sample_peak_dbfs: null, dc_offset: 0, near_full_scale_fraction: 0}],
    arithmetic_mixdown: {complete_cancellation: false, attenuation_db: null},
  };
  context.fixture = {sha256: 'a'.repeat(64), duration: 1, markers: [],
    metrics: {source_measurements: native}, warnings: ['Original-channel warning', '<script>unsafe</script>']};
  vm.runInContext('state.data=fixture; state.file={name:"fixture.wav"}; state.inference={warnings:["Original-channel warning","Model warning"]};', context);
  return {element, run: source => vm.runInContext(source, context), export: () => exported, set: (key, value) => {context[key] = value;}};
}

test('evidence export preserves native measurements and deduplicated signal/model warnings', () => {
  const app = workspace();
  app.element('download-report').onclick();
  const output = app.export();
  assert.equal(output.sha256, 'a'.repeat(64));
  assert.equal(output.metrics.source_measurements.sample_rate_hz, 48000);
  assert.equal(output.metrics.source_measurements.channels[0].rms_dbfs, null);
  assert.deepEqual(output.warnings, ['Original-channel warning', '<script>unsafe</script>', 'Model warning']);
});

test('measurement UI labels source representation, escapes warnings, and clears on reset', () => {
  const app = workspace();
  app.run('renderQuality()');
  assert.match(app.element('source-measurement-summary').textContent, /48,000 Hz/);
  assert.match(app.element('source-measurements').innerHTML, /Undefined \(zero energy\)/);
  assert.match(app.element('source-measurements').innerHTML, /before mixing or resampling/);
  assert.match(app.element('quality-warnings').innerHTML, /&lt;script&gt;/);
  assert.doesNotMatch(app.element('quality-warnings').innerHTML, /<script>/);
  app.run('reset()');
  assert.equal(app.element('quality-warnings').textContent, '');
  assert.equal(app.element('source-measurements').textContent, '');
});

test('assessment leads with verdict and reasons; raw scores stay in collapsed details', () => {
  const app = workspace();
  app.set('result', {assessment: {assessment:'Inconclusive', synthetic_score:null,
    binary_model_decision:null, abstention_reasons:['Input is shorter than 2.5 seconds.', '<unsafe>']},
    model_executed:false, windows:[], links:null, threshold:null});
  app.run('renderAssessment(result)');
  const html = app.element('model-output').innerHTML;
  assert.match(html, /Inconclusive/);
  assert.match(html, /shorter than 2.5 seconds/);
  assert.match(html, /&lt;unsafe&gt;/);
  assert.doesNotMatch(html, /<unsafe>|NaN|0\.000/);
  assert.match(html, /<details><summary>How this result was reached/);
  assert.doesNotMatch(html, /<details open/);
});

test('negative result describes detector evidence without certifying human origin', () => {
  const app = workspace();
  app.set('result', {assessment:{assessment:'Human speech indicated', binary_model_decision:'human',
    synthetic_score:.001, abstention_reasons:[]}, windows:[{}], threshold:.2, links:null});
  app.run('renderAssessment(result)');
  assert.equal(app.element('inference-metric').textContent, 'No clear AI signal');
  assert.match(app.element('model-output').innerHTML, /Short AI inserts can still be missed/);
  assert.doesNotMatch(app.element('model-output').innerHTML, /Human voice indicated|favors human speech/);
});

test('a response for the previous recording cannot overwrite the current assessment', async () => {
  const app = workspace();
  let resolve;
  app.set('FormData', class {append(){}});
  app.set('fetch', () => new Promise(done => {resolve=done;}));
  app.run('state.modelReady=true; state.demo=false;');
  const pending = app.run('checkVoice()');
  app.run('reset(); showAssessmentState("New recording", "Waiting for the new upload");');
  resolve({ok:true, json:async () => ({sha256:'a'.repeat(64), assessment:{assessment:'Human speech indicated'}})});
  await pending;
  assert.match(app.element('model-output').innerHTML, /New recording/);
  assert.doesNotMatch(app.element('model-output').innerHTML, /Human/);
  assert.equal(app.element('run-model').textContent, 'Check voice');
  assert.equal(app.run('state.inference'), null);
});

test('initial screen waits for user recording instead of automatically loading a sample', () => {
  const source = fs.readFileSync(path.join(__dirname, '../studio/dist/app.js'), 'utf8');
  assert.doesNotMatch(source, /;randomStartup\(\);/);
  const html = fs.readFileSync(path.join(__dirname, '../studio/dist/index.html'), 'utf8');
  assert.match(html, /Is this voice AI-generated\?/);
  assert.match(html, /<details class="advanced">/);
});
