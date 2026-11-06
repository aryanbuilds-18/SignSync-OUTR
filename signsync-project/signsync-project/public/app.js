import {HandLandmarker,FilesetResolver} from "/vendor/mediapipe/vision_bundle.mjs";
const WASM="/vendor/mediapipe/wasm";
const MODEL="/models/hand_landmarker.task";
const L=["A","B","C"],ALL=["A","B","C","N"],$=id=>document.getElementById(id);
let lm,stream,data={samples:[],refs:{}},rec=null,msg="",streak=0,cur=null,done=false,lastT=-1,lastRec=0;
try{const r=await fetch("/api/model");if(r.ok){const s=await r.json();if(s&&s.samples)data=s}}catch(e){}
let st;const save=()=>{clearTimeout(st);st=setTimeout(async()=>{try{const r=await fetch("/api/model",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});if(!r.ok)throw 0}catch(e){$("trainMsg").textContent="Could not save to the server. Is it running?"}},400)};

/* tabs */
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".tabs button").forEach(x=>x.setAttribute("aria-selected",x===b));
  document.querySelectorAll(".card section").forEach(s=>s.hidden=s.id!==b.dataset.t);
  if(b.dataset.t==="tts")tiles();
});

/* model */
async function init(src=MODEL){
  $("status").textContent="Loading hand model…";
  try{
    const fs=await FilesetResolver.forVisionTasks(WASM);
    const mk=d=>HandLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:src,delegate:d},runningMode:"IMAGE",numHands:2,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5});
    try{lm=await mk("GPU")}catch(e){lm=await mk("CPU")}
    $("status").textContent="Hand model ready.";
  }catch(e){$("status").textContent="Hand model load nahi hua: "+(e&&e.message||e)+" (model file missing? run: npm run setup)"}
}
init();
$("retry").onclick=()=>init();
$("mfile").onchange=e=>{if(e.target.files[0])init(URL.createObjectURL(e.target.files[0]))};

/* features: up to 2 hands, wrist-relative, scale-normalised */
function feat(res){
  const H=res.landmarks;if(!H||!H.length)return null;
  const hs=[...H].sort((a,b)=>mx(a)-mx(b)).slice(0,2),f=[];let s1=1,w1;
  hs.forEach((h,i)=>{
    const w=h[0];let s=0;h.forEach(p=>s=Math.max(s,Math.hypot(p.x-w.x,p.y-w.y)));s=s||1;
    if(i===0){s1=s;w1=w}
    h.forEach(p=>f.push((p.x-w.x)/s,(p.y-w.y)/s,(p.z-w.z)/s));
  });
  while(f.length<126)f.push(0);
  f.push(hs.length===2?(hs[1][0].x-w1.x)/s1:0,hs.length===2?(hs[1][0].y-w1.y)/s1:0,hs.length*1.5);
  return f.map(v=>+v.toFixed(3));
}
const mx=h=>h.reduce((a,p)=>a+p.x,0)/h.length;

/* classifier: distance-weighted kNN */
function predict(f,set=data.samples.filter(s=>s.set!=="test"),k=5){
  if(!set.length)return null;
  const d=set.map(s=>{let t=0;for(let i=0;i<f.length;i++){const e=f[i]-s.f[i];t+=e*e}return[Math.sqrt(t),s.l]}).sort((a,b)=>a[0]-b[0]).slice(0,k);
  const sc={};ALL.forEach(l=>sc[l]=0);let tot=0;
  d.forEach(([x,l])=>{const w=1/(x+.05);sc[l]+=w;tot+=w});
  ALL.forEach(l=>sc[l]/=tot);
  const best=ALL.reduce((a,b)=>sc[a]>=sc[b]?a:b);
  return{label:best,conf:sc[best],sc};
}


/* built-in ISL prior: geometric rules from the reference signs
   A: two fists, thumbs up, side by side. B: two hands each making a ring. C: one curved hand. */
const REF={A:"/refs/A.png",B:"/refs/B.png",C:"/refs/C.png"};
function hand(f,i){const o=i*63,P=[];for(let j=0;j<21;j++)P.push([f[o+j*3],f[o+j*3+1]]);return P}
const dd=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
function hinfo(P){
  const w=P[0],ext=[8,12,16,20].map((t,k)=>dd(P[t],w)/(dd(P[5+4*k],w)||1));
  return{ext:ext.reduce((a,b)=>a+b,0)/4,ring:dd(P[4],P[8]),thumbOut:dd(P[4],P[5])};
}
function rules(f){
  const n=Math.round(f[128]/1.5);if(!n)return null;
  const h=[];for(let i=0;i<n;i++)h.push(hinfo(hand(f,i)));
  let l=null;
  if(n===1){const a=h[0];if(a.ring>.3&&a.ring<1&&a.ext>1.15&&a.ext<1.95)l="C"}
  else{
    if(h.every(a=>a.ring<.35))l="B";
    else if(h.every(a=>a.ext<1.5&&a.thumbOut>.55))l="A";
  }
  if(!l)return null;
  const sc={};L.forEach(x=>sc[x]=x===l?.85:.075);return{label:l,conf:.85,sc};
}
function classify(f){
  const r=rules(f),set=data.samples.filter(s=>s.set!=="test"),k=set.length?predict(f):null;
  if(k&&k.label==="N")return k;if(!r)return k;if(!k)return r;
  const sc={};L.forEach(x=>sc[x]=.65*k.sc[x]+.35*r.sc[x]);
  const b=L.reduce((a,c)=>sc[a]>=sc[c]?a:c);return{label:b,conf:sc[b],sc};
}

/* skeleton */
const CON=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const cv=$("c"),cx=cv.getContext("2d"),v=$("v");
function draw(H){
  cv.width=v.videoWidth||640;cv.height=v.videoHeight||480;cx.clearRect(0,0,cv.width,cv.height);
  (H||[]).forEach(h=>{
    cx.strokeStyle="#35c79a";cx.lineWidth=3;
    CON.forEach(([a,b])=>{cx.beginPath();cx.moveTo(h[a].x*cv.width,h[a].y*cv.height);cx.lineTo(h[b].x*cv.width,h[b].y*cv.height);cx.stroke()});
    cx.fillStyle="#ff8a5c";h.forEach(p=>{cx.beginPath();cx.arc(p.x*cv.width,p.y*cv.height,4,0,7);cx.fill()});
  });
}

/* camera */
$("camBtn").onclick=async()=>{
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;v.hidden=true;$("ph").hidden=false;$("camBtn").textContent="Start camera";return}
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:"user"}});
    v.srcObject=stream;await v.play();v.hidden=false;$("ph").hidden=true;$("camBtn").textContent="Stop camera";loop();
  }catch(e){$("status").textContent="Camera error: "+(e&&e.name)+" - browser me camera Allow karo, aur koi dusra app camera use na kar raha ho."}
};
function loop(){
  if(!stream)return;
  if(lm&&v.currentTime!==lastT&&v.readyState>=2){
    lastT=v.currentTime;
    const res=lm.detect(v);draw(res.landmarks);
    const f=feat(res);
    if(rec&&f&&performance.now()-lastRec>90){lastRec=performance.now();addSample(f,rec.l,rec.set);rec.n--;
      if(rec.n===1||!data.refs[rec.l])snap(rec.l);
      $("rec").textContent=`Recording ${rec.l}… ${rec.n} left`;
      if(rec.n<=0){rec=null;$("rec").textContent="Record 60 samples";save();counts();$("trainMsg").textContent="Done."}}
    live(f);
  }
  requestAnimationFrame(loop);
}
function snap(l){const t=document.createElement("canvas");t.width=t.height=160;const g=t.getContext("2d");
  const s=Math.min(v.videoWidth,v.videoHeight);g.translate(160,0);g.scale(-1,1);
  g.drawImage(v,(v.videoWidth-s)/2,(v.videoHeight-s)/2,s,s,0,0,160,160);data.refs[l]=t.toDataURL("image/jpeg",.7)}

/* live prediction + sentence builder */
function live(f){
  const thr=$("thr").value/100;let p=f&&classify(f);
  const ok=p&&p.label!=="N"&&p.conf>=thr;
  $("badge").textContent=ok?p.label:"–";$("pred").textContent=ok?p.label:"–";
  $("confBar").style.width=p?p.conf*100+"%":"0";$("confTxt").textContent=p?Math.round(p.conf*100)+"% "+(ok?"":p.label==="N"?"(not a sign)":"(below threshold)"):"No hand";
  $("cls").innerHTML=L.map(l=>`<div class="cls"><b>${l}</b><div class="bar"><i style="width:${p?p.sc[l]*100:0}%"></i></div><span>${p?Math.round(p.sc[l]*100):0}%</span></div>`).join("");
  const lab=ok?p.label:null;
  if(lab&&lab===cur)streak++;else{streak=lab?1:0;cur=lab;done=false}
  if(lab&&streak>=12&&!done){done=true;msg+=lab;showMsg();if($("auto").checked)say(lab)}
}
const showMsg=()=>$("msg").textContent=msg;
$("thr").oninput=e=>$("thrV").textContent=e.target.value+"%";
$("sp").onclick=()=>{msg+=" ";showMsg()};$("bk").onclick=()=>{msg=msg.slice(0,-1);showMsg()};$("clr").onclick=()=>{msg="";showMsg()};
const say=t=>{try{speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(t.split("").join(" ")))}catch(e){}};
$("spk").onclick=()=>say(msg||"Nothing to speak");

/* training */
function addSample(f,l,set){data.samples.push({f,l,set})}
function counts(){$("counts").innerHTML=ALL.map(l=>{const a=data.samples.filter(s=>s.l===l);return`<span>${l}: ${a.filter(s=>s.set!=="test").length} train / ${a.filter(s=>s.set==="test").length} test</span>`}).join("")}
counts();
$("rec").onclick=()=>{
  if(!stream||!lm){$("trainMsg").textContent="Start the camera first.";return}
  rec={l:$("lab").value,set:$("setSel").value,n:60};$("trainMsg").textContent="Hold the sign and move slightly for variety.";
};
$("up").onchange=async e=>{
  if(!lm){$("trainMsg").textContent="Model not ready.";return}
  let ok=0,bad=0;const l=$("lab").value,set=$("setSel").value;
  for(const file of e.target.files){
    try{
      const img=new Image();img.src=URL.createObjectURL(file);await img.decode();
      const f=feat(lm.detect(img));
      if(f){addSample(f,l,set);ok++;if(!data.refs[l]){const t=document.createElement("canvas");t.width=t.height=160;t.getContext("2d").drawImage(img,0,0,160,160);data.refs[l]=t.toDataURL("image/jpeg",.7)}}else bad++;
    }catch(x){bad++}
  }
  save();counts();$("trainMsg").textContent=`Added ${ok} images to ${l}; ${bad} had no detectable hand.`;e.target.value="";
};
$("exp").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:"application/json"}));a.download="signsync-model.json";a.click()};
$("imp").onchange=async e=>{try{data=JSON.parse(await e.target.files[0].text());save();counts();$("trainMsg").textContent="Model imported."}catch(x){$("trainMsg").textContent="That file is not a SignSync model."}};
$("wipe").onclick=()=>{if(confirm("Delete all recorded samples?")){data={samples:[],refs:{}};save();counts()}};


$("wiz").onclick=async()=>{
  if(!stream||!lm){$("trainMsg").textContent="Pehle camera start karo aur model ready hone do.";return}
  if(data.samples.length&&!confirm("Purane samples delete karke naye record karein?"))return;
  data.samples=[];counts();
  const wait=ms=>new Promise(r=>setTimeout(r,ms)),P=$("prompt");P.hidden=false;
  const plan=[...ALL.map(l=>[l,"train"]),...ALL.map(l=>[l,"test"])];
  for(const [l,set] of plan){
    const hint=set==="train"?"Training":"Held-out test: thoda alag doori/position/roshni rakho";
    const nm=l==="N"?"OTHER: koi bhi random haath pose (khula haath, mutthi, ungli, peace) badalte raho":l+" sign banao";
    for(let c=5;c>0;c--){P.textContent=`${hint} · ${nm}: ${c}`;await wait(1000)}
    rec={l,set,n:60};
    while(rec){P.textContent=`${set==="train"?"Training":"Test"} ${l}: sign hold karo aur thoda hilao (${rec.n} baaki)`;await wait(200)}
  }
  P.hidden=true;save();
  document.querySelector('[data-t="eval"]').click();$("runEval").click();
};

/* text/speech to sign */
function tiles(){
  const t=$("tin").value.toUpperCase().replace(/[^A-Z ]/g,"");
  $("tiles").innerHTML=[...t].filter(c=>c!==" ").map(c=>{
    if(!L.includes(c))return`<div class="tile x"><div class="ph2">outside vocabulary</div><b>${c}</b></div>`;
    return`<div class="tile">${(REF[c]||data.refs[c])?`<img src="${REF[c]||data.refs[c]}" alt="ISL sign for ${c}">`:`<div class="ph2">no reference yet</div>`}<b>${c}</b></div>`}).join("");
}
$("tin").oninput=tiles;
$("mic").onclick=()=>{
  const R=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!R){$("mic").textContent="Mic not supported here";return}
  const r=new R();r.lang="en-IN";r.onresult=e=>{$("tin").value=e.results[0][0].transcript.replace(/[^a-zA-Z ]/g,"").slice(0,24);tiles()};r.start();
};

/* evaluation */
$("runEval").onclick=()=>{
  let tr=data.samples.filter(s=>s.set!=="test"),te=data.samples.filter(s=>s.set==="test"),note="";
  if(!te.length){
    const sh=[...data.samples].sort(()=>Math.random()-.5);const n=Math.floor(sh.length*.2);
    te=sh.slice(0,n);tr=sh.slice(n);note="No held-out test set recorded, so a random 20% split was used. Neighbouring video frames are near-duplicates, so this score is optimistic. Record a separate session as held-out test.";
  }
  if(!tr.length||!te.length||L.some(l=>!tr.some(s=>s.l===l))){$("evalOut").textContent="Record training samples for A, B and C first.";return}
  const M={};ALL.forEach(a=>{M[a]={};ALL.forEach(b=>M[a][b]=0)});let ok=0,cs=0;
  te.forEach(s=>{const p=predict(s.f,tr);M[s.l][p.label]++;if(p.label===s.l){ok++;cs+=p.conf}});
  const acc=ok/te.length;
  fetch("/api/report",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({time:new Date().toISOString(),accuracy:acc,testSamples:te.length,heldOutSession:!note,confusion:M})}).catch(()=>{});
  $("evalOut").innerHTML=`<div class="cnt"><span>Accuracy: ${(acc*100).toFixed(1)}%</span><span>Held-out samples: ${te.length}</span><span>Mean confidence on correct: ${ok?Math.round(cs/ok*100):0}%</span></div>
  <table><tr><th>true \ predicted</th>${ALL.map(l=>`<th>${l}</th>`).join("")}</tr>${ALL.map(a=>`<tr><th>${a}</th>${ALL.map(b=>`<td${a===b?' style="font-weight:700"':""}>${M[a][b]}</td>`).join("")}</tr>`).join("")}</table>
  ${note?`<div class="warn">${note}</div>`:""}`;
};