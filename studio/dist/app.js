"use strict";
const $=id=>document.getElementById(id);
const state={data:null,inference:null,file:null,url:null,selected:null,filter:"all",zoom:1,offset:0,loop:false,benchmark:null,modelReady:false,notes:{},demo:false,generation:0,busy:false,statusError:false};
let statusPromise;
const COLORS={signal:"#6bddc9",model:"#bc96f3",reliability:"#f0b86e"};

let toastTimer;
function toast(msg){const n=$("toast");n.textContent=msg;n.classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>n.classList.remove("show"),4500)}
function clock(n){if(!Number.isFinite(n))return "00:00";return String(Math.floor(n/60)).padStart(2,"0")+":"+String(Math.floor(n%60)).padStart(2,"0")}
function precise(n){return Number(n).toFixed(2)+"s"}
function escapeHTML(text){return String(text??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function markers(){return [...(state.data?.markers||[]),...(state.inference?.markers||[])].sort((a,b)=>a.start-b.start)}
function filtered(){return markers().filter(m=>state.filter==="all"||m.category===state.filter||(state.filter==="model"&&m.category==="reliability"))}
function setPage(page){for(const n of document.querySelectorAll(".nav"))n.classList.toggle("active",n.dataset.page===page);$("analyze").classList.toggle("hidden",page!=="analyze");$("benchmark").classList.toggle("hidden",page!=="benchmark");$("breadcrumb").textContent=page==="analyze"?"VOICE CHECK":"VALIDATION HISTORY";if(page==="benchmark")loadBenchmark();if(page==="analyze")requestAnimationFrame(draw)}
document.querySelectorAll(".nav").forEach(b=>b.addEventListener("click",()=>setPage(b.dataset.page)));
function resize(){const canvas=$("wave");const width=canvas.parentElement.clientWidth;if(!width)return;const dpr=Math.min(2,devicePixelRatio||1);canvas.width=width*dpr;canvas.height=242*dpr;canvas.getContext("2d").setTransform(dpr,0,0,dpr,0,0);draw()}
window.addEventListener("resize",resize);
function draw(){const canvas=$("wave");if(!canvas.width)return;const ctx=canvas.getContext("2d"),w=canvas.clientWidth,h=242;ctx.clearRect(0,0,w,h);ctx.fillStyle="#0b1928";ctx.fillRect(0,0,w,h);const duration=state.data?.duration||1;const view=duration/state.zoom;const begin=state.offset;const xAt=t=>(t-begin)/view*w;
for(let i=0;i<=6;i++){let y=22+i*(h-44)/6;ctx.strokeStyle="#19354a";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
if(!state.data)return;
const all=markers();for(const m of all){if(m.end<begin||m.start>begin+view)continue;const x=Math.max(0,xAt(m.start)),right=Math.min(w,xAt(m.end));ctx.fillStyle=m.category==="signal"?"rgba(103,215,196,.15)":m.category==="model"?"rgba(181,146,236,.18)":"rgba(234,178,91,.18)";ctx.fillRect(x,16,Math.max(2,right-x),h-32);ctx.fillStyle=COLORS[m.category]||"#ecc178";ctx.fillRect(x,15,Math.max(2,right-x),3)}
const peaks=state.data.peaks;
ctx.strokeStyle="#5fd1be";ctx.lineWidth=1.3;ctx.beginPath();for(let x=0;x<w;x+=1.8){const t=begin+(x/w)*view;const idx=Math.max(0,Math.min(peaks.length-1,Math.round(t/duration*(peaks.length-1))));const p=peaks[idx]||[0,0];ctx.moveTo(x,h/2-(p[1]||0)*87);ctx.lineTo(x,h/2-(p[0]||0)*87)}ctx.stroke();
if(state.selected){const m=all.find(m=>m.id===state.selected);if(m){ctx.strokeStyle=COLORS[m.category]||"#fff";ctx.lineWidth=2;ctx.strokeRect(Math.max(1,xAt(m.start)),17,Math.max(2,(m.end-m.start)/view*w),h-34)}}
const now=$("player").currentTime;if(now>=begin&&now<=begin+view){const x=xAt(now);ctx.strokeStyle="#f4fff5";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,7);ctx.lineTo(x,h-7);ctx.stroke()}
ctx.fillStyle="#7596ab";ctx.font="11px system-ui";ctx.textAlign="left";for(let i=0;i<=5;i++){const t=begin+i/5*view;ctx.fillText(clock(t),Math.min(w-36,i/5*w)+6,h-3)}
}
function seek(t){const p=$("player");p.currentTime=Math.max(0,Math.min(state.data.duration,t));draw()}
$("wave").addEventListener("click",event=>{if(!state.data)return;const rect=event.target.getBoundingClientRect();const t=state.offset+(event.clientX-rect.left)/rect.width*state.data.duration/state.zoom;const nearby=markers().filter(m=>t>=m.start&&t<=m.end).sort((a,b)=>(a.end-a.start)-(b.end-b.start));if(nearby.length)selectMarker(nearby[0].id);else seek(t)});
$("wave").addEventListener("wheel",event=>{if(!state.data)return;event.preventDefault();zoom(event.deltaY<0?1.35:1/1.35)},{passive:false});
function zoom(factor){if(!state.data)return;const duration=state.data.duration;const midpoint=state.offset+duration/state.zoom/2;state.zoom=Math.max(1,Math.min(16,state.zoom*factor));state.offset=Math.max(0,Math.min(duration-duration/state.zoom,midpoint-duration/state.zoom/2));$("zoom-label").textContent=state.zoom.toFixed(1)+"×";draw()}
$("zoom-in").onclick=()=>zoom(1.6);$("zoom-out").onclick=()=>zoom(1/1.6);
$("progress").addEventListener("click",e=>{if(!state.data)return;const r=e.currentTarget.getBoundingClientRect();seek((e.clientX-r.left)/r.width*state.data.duration)});
$("play").addEventListener("click",()=>{const p=$("player");if(p.paused)p.play().catch(()=>toast("Could not play this recording."));else p.pause()});
$("player").addEventListener("play",()=>{$("play").textContent="❚❚"});
$("player").addEventListener("pause",()=>{$("play").textContent="▶"});
$("player").addEventListener("timeupdate",()=>{if(!state.data)return;const p=$("player");$("current-time").textContent=clock(p.currentTime);$("progress-fill").style.width=(100*p.currentTime/state.data.duration)+"%";if(state.loop&&state.selected){const m=markers().find(x=>x.id===state.selected);if(m&&p.currentTime>=m.end)p.currentTime=m.start}draw()});
$("loop").onclick=()=>{state.loop=!state.loop;$("loop").textContent=state.loop?"↻ Loop on":"↻ Loop off";toast(state.loop?"Selected interval will loop.":"Loop disabled.")};
document.addEventListener("keydown",e=>{if(e.key!==" "||["INPUT","TEXTAREA","BUTTON","SELECT"].includes(document.activeElement.tagName)||!state.data)return;e.preventDefault();$("play").click()});
const ICON_PATHS={
 clipping:'<path d="M12 3v11m0 4v3M5 12h3m8 0h3"/>',
 low_energy:'<path d="M3 12h5l2-3 4 6 2-3h5"/>',
 level_change:'<path d="M3 19V7h7v8h11M15 4l6 3-6 3"/>',
 spectral_change:'<path d="M3 19V5M7 18v-5m4 5V8m4 10V4m4 14v-9M3 19h18"/>',
 signal_gap:'<path d="M2 12h6m8 0h6M9 6v12m6-12v12"/>',
 impulse:'<path d="M2 12h7l2-9 3 18 2-9h6"/>',
 dc_offset:'<path d="M3 17h18M3 6h18M7 12c3-7 7-7 10 0"/>',
 bandwidth_limit:'<path d="M3 19V5m0 14h18M6 16l4-5 4-3h5m-5-5v16" stroke-dasharray="2 1"/>',
 channel_imbalance:'<path d="M4 7h12M4 12h7M4 17h16M19 4v6"/>',
 phase_inversion:'<path d="M2 8c4-8 7 8 11 0s7-8 9 0M2 16c4 8 7-8 11 0s7 8 9 0"/>',
 synthetic_score:'<path d="M12 2l8 5v10l-8 5-8-5V7zM9 12l2 2 4-5"/>',
 component_separation:'<circle cx="7" cy="12" r="4"/><circle cx="17" cy="12" r="4"/><path d="M11 12h2"/>',
 sensitivity:'<path d="M3 7h18l-3-3m3 3l-3 3M21 17H3l3-3m-3 3l3 3"/>'
};
function evidenceIcon(kind){const path=ICON_PATHS[kind]||'<circle cx="12" cy="12" r="8"/>';return '<svg class="evidence-svg" viewBox="0 0 24 24" width="23" height="23" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+path+'</svg>'}
function renderMarkers(){const list=$("markers");const items=filtered();$("count").textContent=markers().length;$("marker-metric").textContent=markers().length;if(!items.length){list.innerHTML='<div class="blank"><span>⌁</span><strong>No matching evidence</strong><p>Try another filter or upload a recording.</p></div>';return}list.innerHTML=items.map((m,i)=>'<button class="marker-card '+(m.category==="model"?"model":"")+(state.selected===m.id?" active":"")+'" data-marker="'+escapeHTML(m.id)+'"><span class="marker-icon">'+evidenceIcon(m.kind)+'</span><span class="marker-copy"><strong>'+escapeHTML(m.title)+'</strong><small>'+escapeHTML(m.description)+'</small></span><span class="marker-time">'+precise(m.start)+'–'+precise(m.end)+'</span></button>').join("");list.querySelectorAll("[data-marker]").forEach(b=>b.onclick=()=>selectMarker(b.dataset.marker))}
function selectMarker(id){const m=markers().find(x=>x.id===id);if(!m)return;state.selected=id;const currentView=state.data.duration/state.zoom;if(m.start<state.offset||m.end>state.offset+currentView)state.offset=Math.max(0,Math.min(state.data.duration-currentView,m.start-currentView*.18));$("marker-category").textContent=(m.category==="signal"?"ACOUSTIC EVIDENCE":m.category==="model"?"MODEL OUTPUT":"RELIABILITY")+" · "+precise(m.start)+"–"+precise(m.end);
let facts=Object.entries(m.measurements||{}).map(([k,v])=>'<div class="measurement"><span>'+escapeHTML(k.replace(/_/g," "))+'</span><strong>'+escapeHTML(v)+'</strong></div>').join("");
$("detail").innerHTML='<h3 class="detail-header">'+escapeHTML(m.title)+'</h3><p class="detail-copy">'+escapeHTML(m.description)+'</p><div class="fact"><b>MEASURED OBSERVATIONS</b>'+facts+'</div><div class="fact"><b>HOW IT WAS DETECTED</b><p>'+escapeHTML(m.method)+'</p><p>Rule: '+escapeHTML(m.threshold)+'</p></div><div class="fact"><b>WHAT THIS DOES NOT PROVE</b><p>'+escapeHTML(m.limitations)+'</p></div><div class="fact"><b>RECOMMENDED CHECK</b><p>'+escapeHTML(m.next_check)+'</p></div><div class="fact"><b>ANALYST NOTE (LOCAL SESSION)</b><textarea id="analyst-note" style="width:100%;min-height:78px;background:#0f2031;border:1px solid #39546a;border-radius:8px;padding:10px;color:white" placeholder="Record your observation; included in evidence export.">'+escapeHTML(state.notes[id]||"")+'</textarea></div>';
$("analyst-note").addEventListener("input",e=>{state.notes[id]=e.target.value});
$("replay").disabled=false;renderMarkers();draw();}
$("replay").onclick=()=>{const m=markers().find(x=>x.id===state.selected);if(!m)return;state.loop=false;$("loop").textContent="↻ Loop off";seek(Math.max(0,m.start-.4));$("player").play().catch(()=>toast("Playback failed."))};
document.querySelectorAll("[data-filter]").forEach(b=>b.onclick=()=>{state.filter=b.dataset.filter;document.querySelectorAll("[data-filter]").forEach(x=>x.classList.toggle("selected",x===b));renderMarkers()});
function renderQuality(){
  const data=state.data;
  $("recording-quality").classList.toggle("hidden",!data);
  if(!data)return;
  const warnings=[...new Set([...(data.warnings||[]),...(state.inference?.warnings||[])])];
  $("quality-warnings").innerHTML=warnings.length?'<ul>'+warnings.map(w=>'<li>'+escapeHTML(w)+'</li>').join('')+'</ul>':'<p>No configured quality warning triggered. This does not establish authenticity.</p>';
  const source=data.metrics.source_measurements;
  if(!source){$("source-measurement-summary").textContent="Source measurements unavailable";$("source-measurements").textContent="";return;}
  const db=value=>value===null?"Undefined (zero energy)":Number(value).toFixed(2);
  $("source-measurement-summary").textContent=source.sample_rate_hz.toLocaleString()+" Hz · "+source.channels.length+" original channel(s) · "+source.version;
  const rows=source.channels.map(c=>'<tr><td>'+c.channel+'</td><td>'+db(c.rms_dbfs)+'</td><td>'+db(c.sample_peak_dbfs)+'</td><td>'+Number(c.dc_offset).toPrecision(4)+'</td><td>'+(100*c.near_full_scale_fraction).toFixed(3)+'%</td></tr>').join('');
  const mix=source.arithmetic_mixdown;
  const mixText=mix.complete_cancellation?"Original channels cancel completely in the arithmetic mono mix.":mix.attenuation_db===null?"Mixdown attenuation is undefined for a silent source.":"Arithmetic mono mix attenuation: "+Number(mix.attenuation_db).toFixed(2)+" dB relative to mean channel power.";
  $("source-measurements").innerHTML='<div class="measurement-table"><table><thead><tr><th>Channel</th><th>RMS (dBFS)</th><th>Sample peak (dBFS)</th><th>DC offset</th><th>Near full scale</th></tr></thead><tbody>'+rows+'</tbody></table></div><p>'+escapeHTML(mixText)+'</p><p>These measurements use decoded source channels before mixing or resampling. Near full scale means |sample| ≥ '+source.near_full_scale_threshold+'; this does not prove clipping. Sample peaks are not reconstructed true peaks.</p><p>The waveform, acoustic markers and summary RMS use an arithmetic mono mix resampled to 16,000 Hz. Channel cancellation and source frequencies above 8 kHz can be hidden in that view.</p>';
}
function reset(){state.generation++;state.busy=false;state.file=null;state.inference=null;state.data=null;state.selected=null;state.notes={};state.zoom=1;state.offset=0;state.loop=false;$("recording-quality").classList.add("hidden");$("quality-warnings").textContent="";$("source-measurements").textContent="";$("zoom-label").textContent="1.0×";showAssessmentState("Ready when you are.","Upload a voice recording to check it.");$("inference-metric").textContent="Not run";$("detail").innerHTML='<div class="blank"><span>◇</span><strong>Select a highlighted interval</strong><p>Click any marker to see the underlying observation and its limitations.</p></div>';$("marker-category").textContent="NO SELECTION";$("download-report").disabled=true;$("run-model").disabled=true;$("run-model").textContent="Check voice";$("replay").disabled=true;$("play").disabled=true;$("loop").disabled=true;renderMarkers();draw()}
function showAssessmentState(title, description){$("model-output").innerHTML='<div class="assessment-empty"><span class="eyebrow">ASSESSMENT</span><h2>'+escapeHTML(title)+'</h2><p>'+escapeHTML(description)+'</p></div>'}
function renderAssessment(payload){
  const a=payload.assessment;
  const inconclusive=a.assessment==="Inconclusive";
  const kind=inconclusive?"inconclusive":a.binary_model_decision==="synthetic"?"synthetic":"human";
  const title=inconclusive?"Inconclusive":kind==="synthetic"?"AI voice indicated":"No clear AI signal";
  const description=inconclusive?"There is not enough reliable evidence for a clear result.":kind==="synthetic"?"The detector found patterns associated with AI-generated speech.":"The detector did not find a clear AI signal. Short AI inserts can still be missed.";
  $("inference-metric").textContent=title;
  const score=typeof a.synthetic_score==="number"?'<div class="scores"><div class="score"><span>RAW SYNTHETIC SCORE</span><strong>'+a.synthetic_score.toFixed(3)+'</strong></div><div class="score"><span>ANALYZED WINDOWS</span><strong>'+payload.windows.length+'</strong></div><div class="score"><span>DECISION THRESHOLD</span><strong>'+Number(payload.threshold).toFixed(3)+'</strong></div></div><p>Raw scores are not calibrated probabilities of authenticity.</p>':'<p>Model inference was skipped because the recording failed quality checks.</p>';
  const links=payload.links?'<p><a href="'+escapeHTML(payload.links.html)+'">Download detailed report</a> · <a href="'+escapeHTML(payload.links.json)+'">JSON</a></p>':'';
  $("model-output").innerHTML='<div class="model-output" data-result="'+kind+'"><span class="eyebrow">ASSESSMENT</span><h2 class="assessment-title">'+title+'</h2><p class="assessment-note">'+description+'</p>'+((a.abstention_reasons||[]).length?'<ul class="assessment-reasons">'+a.abstention_reasons.map(r=>'<li>'+escapeHTML(r)+'</li>').join('')+'</ul>':'')+'<p>An indication, not proof. Results are best supported on English speech.</p><details><summary>How this result was reached</summary>'+score+links+'</details></div>';
}
async function loadFile(file,demo){
  reset();const generation=state.generation;state.file=file;state.demo=demo;
  if(state.url)URL.revokeObjectURL(state.url);state.url=URL.createObjectURL(file);$("player").src=state.url;
  $("filename").textContent=file.name;$("source-status").textContent="Preparing your recording locally…";
  $("empty-wave").classList.add("hidden");showAssessmentState("Preparing recording…","Checking audio quality before detection.");
  try{
    const body=new FormData();body.append("file",file);
    const response=await fetch("/api/inspect",{method:"POST",body});const payload=await response.json();
    if(generation!==state.generation)return;
    if(!response.ok)throw Error(payload.error||"Audio analysis failed");
    state.data=payload;$("source-status").textContent=(demo?"Generated non-speech demo · Voice detection is disabled.":"Recording ready · "+payload.duration.toFixed(1)+" seconds · Processed locally");
    $("duration").textContent=clock(payload.duration);$("time-end").textContent=clock(payload.duration);
    $("duration-metric").textContent=payload.duration.toFixed(1)+"s";$("rms-metric").textContent=payload.metrics.rms_dbfs.toFixed(1)+" dBFS";
    $("play").disabled=false;$("loop").disabled=false;$("download-report").disabled=false;
    renderQuality();renderMarkers();resize();
    await statusPromise;if(generation!==state.generation)return;
    $("run-model").disabled=!state.modelReady||demo;
    if(demo)showAssessmentState("Signal demonstration","This is a generated non-speech fixture, not an AI-voice detection example.");
    else if(state.modelReady)await checkVoice();
    else showAssessmentState("Detector unavailable",state.statusError?"The local service could not be reached. Refresh to try again.":"Set up the local voice detector to assess this recording.");
  }catch(e){if(generation!==state.generation)return;$("source-status").textContent=e.message;$("empty-wave").classList.remove("hidden");showAssessmentState("Recording could not be checked",e.message);toast(e.message)}
}

$("upload-button").onclick=()=>$("audio-file").click();$("audio-file").addEventListener("change",e=>{const file=e.target.files[0];if(file){loadFile(file,false);e.target.value=""}});
async function loadCatalogSample(entry){
  if(!entry)throw Error("No startup sample found");
  const response=await fetch("/api/samples/"+encodeURIComponent(entry.id)+"/audio");
  if(!response.ok)throw Error("Selected sample is no longer available");
  const blob=await response.blob();
  const ext=(entry.extension||".wav").replace(/^\./,"");
  const sample=new File([blob],entry.label+"."+ext,{type:blob.type==="application/octet-stream"?"":blob.type});
  await loadFile(sample,entry.source==="generated_signal");
  if(state.data&&state.file===sample){$("source-status").textContent=(entry.source==="generated_signal"?"GENERATED NON-SPEECH SIGNAL · ":"LOCAL RECORDING DATABASE · ")+"Measured evidence · "+state.data.duration.toFixed(1)+" seconds";state.sampleSource=entry.source;}
}
async function randomStartup(){
  try{
    $("source-status").textContent="Opening a random local sample…";
    const response=await fetch("/api/samples/random");
    if(!response.ok)throw Error("Startup sample catalog unavailable");
    await loadCatalogSample(await response.json());
  }catch(error){
    $("source-status").textContent="No startup sample could be opened. Upload a recording or use the signal demo.";
    toast(error.message);
  }
}
$("random-button").onclick=randomStartup;
$("demo-button").onclick=async()=>{try{toast("Loading measured signal demonstration…");const r=await fetch("/api/demo");if(!r.ok)throw Error("Example not available");const b=await r.blob();await loadFile(new File([b],"signal-test-tone.wav",{type:"audio/wav"}),true)}catch(e){toast(e.message)}};
async function checkVoice(){
  if(!state.file||!state.data||!state.modelReady||state.demo||state.busy)return;
  const generation=state.generation,original=state.data.sha256,file=state.file,button=$("run-model");
  state.busy=true;button.disabled=true;button.textContent="Checking voice…";
  state.inference=null;$("inference-metric").textContent="Checking…";renderQuality();renderMarkers();
  showAssessmentState("Checking voice…","The detector runs locally. Longer recordings take more time.");
  try{
    const body=new FormData();body.append("file",file);
    const response=await fetch("/api/infer",{method:"POST",body});const payload=await response.json();
    if(generation!==state.generation)return;
    if(!response.ok)throw Error(payload.error||"Inference failed");
    if(payload.sha256!==original)throw Error("File identity changed; result rejected.");
    state.inference=payload;renderAssessment(payload);renderQuality();renderMarkers();draw();
    $("download-report").disabled=false;
  }catch(e){if(generation!==state.generation)return;showAssessmentState("Assessment unavailable",e.message);$("inference-metric").textContent="Not available";toast(e.message)}
  finally{if(generation===state.generation){state.busy=false;button.textContent="Check voice";button.disabled=!state.modelReady||state.demo}}
}
$("run-model").onclick=checkVoice;
$("download-report").onclick=()=>{if(!state.data)return;const payload={product:"SignalTrace",type:"recording_evidence",generated_at:new Date().toISOString(),file_name:state.file.name,sha256:state.data.sha256,duration_seconds:state.data.duration,metrics:state.data.metrics,warnings:[...new Set([...(state.data.warnings||[]),...(state.inference?.warnings||[])])],markers:markers().map(m=>({...m,analyst_note:state.notes[m.id]||""})),assessment:state.inference?.assessment||null,model_executed:state.inference?.model_executed??null,model_report:state.inference?.links||null,limitations:["Signal markers are measurements, not verified edits.","Model scores are not calibrated authenticity probabilities.","The supplied signal demo is synthetic non-speech test tone."]};const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const anchor=document.createElement("a");anchor.href=url;anchor.download="signaltrace-evidence.json";anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
async function loadStatus(){try{
  const r=await fetch("/api/status");if(!r.ok)throw Error("Detector status unavailable");const status=await r.json();
  state.modelReady=status.ready;state.statusError=false;
  $("model-status").textContent=status.ready?"● Voice detector ready":"○ Detector setup required";
  $("model-status").title=status.ready?"Local model checkpoints available":"Install checkpoints to assess voices";
  $("setup").classList.toggle("hidden",status.ready);$("setup-commands").textContent=status.setup.join("\n");
  $("run-model").disabled=!status.ready||!state.data||state.demo||state.busy;
  if(!status.ready)showAssessmentState("Set up voice detection","The local model checkpoints are required. Use the setup instructions below.");
}catch(e){state.modelReady=false;state.statusError=true;$("model-status").textContent="Service unavailable";showAssessmentState("Service unavailable","Start the local server and refresh this page.")}}

$("setup").onclick=()=>$("setup-dialog").showModal();$("close-setup").onclick=()=>$("setup-dialog").close();
async function loadBenchmark(){if(state.benchmark){renderBenchmark();return}try{const r=await fetch("/api/benchmark");const result=await r.json();if(!r.ok)throw Error(result.error||"Benchmark unavailable");state.benchmark=result;renderBenchmark()}catch(e){toast(e.message)}}
function renderBenchmark(){const all=state.benchmark;if(!all)return;const condition=$("condition").value,model=$("model-choice").value;const values=all.conditions[condition][model];const fmt=(n,d)=>d?((100*n/d).toFixed(1)+"%"):"—";$("bench-synthetic").textContent=fmt(values.synthetic_flagged,values.synthetic_total);$("bench-human").textContent=fmt(values.human_flagged,values.human_total);$("bench-synthetic-count").textContent=values.synthetic_flagged+" of "+values.synthetic_total+" synthetic recordings";$("bench-human-count").textContent=values.human_flagged+" of "+values.human_total+" genuine recordings";const names={"fusion":"Two-model fusion","pellav2 alone (same rule)":"PellaV2 alone","AntiDeepfake alone (same rule)":"AntiDeepfake alone","pellav2 at authors’ 0.5":"PellaV2 default threshold"};$("comparison").innerHTML=Object.entries(all.conditions[condition]).map(([name,v])=>'<div class="compare-row"><span>'+escapeHTML(names[name]||name)+'</span><div class="bar"><div class="bar-inner" style="width:'+Math.min(100,100*v.human_flagged/v.human_total)+'%"></div></div><span>'+v.human_flagged+"/"+v.human_total+" false alerts</span></div>").join('')+'<p class="hint">Bars compare genuine recordings falsely flagged. Lower is better. Threshold selection differs by model.</p>';$("limitations").innerHTML=all.limitations.map(x=>"<p>• "+escapeHTML(x)+"</p>").join("");}
$("condition").onchange=renderBenchmark;$("model-choice").onchange=renderBenchmark;
statusPromise=loadStatus();window.addEventListener("load",resize);setTimeout(resize,100);
document.querySelectorAll(".playback-details,.advanced").forEach(detail=>detail.addEventListener("toggle",resize));
