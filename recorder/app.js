const SERVICE_UUID = "4798e0f2-0000-4d68-af64-8a8f5258404e";
const STROKE_UUID = "4798e0f2-300a-4d68-af64-8a8f5258404e";
const labels = ["wing", "ring", "slope", "unknown"];
const state = { characteristic: null, timer: null, strokes: [], lastDone: "" };
const $ = (id) => document.getElementById(id);
const status = (message) => { $("status").textContent = message; };
const sessionId = () => `${$("label").value}-${$("split").value}-${new Date().toISOString().slice(0,16).replace(/[:T]/g,"-")}`;

function parseStroke(view) {
  if (view.byteLength < 8) return null;
  const gestureState = view.getInt32(0, true);
  const length = Math.max(0, Math.min(160, view.getInt32(4, true)));
  const points = [];
  for (let index = 0; index < length && 9 + index * 2 < view.byteLength; index++) {
    points.push({ x: view.getInt8(8 + index * 2) / 128, y: view.getInt8(9 + index * 2) / 128, t: length > 1 ? index / (length - 1) : 0 });
  }
  return { gestureState, points };
}

function draw(canvas, points) {
  const context = canvas.getContext("2d"); context.fillStyle = "#071321"; context.fillRect(0,0,canvas.width,canvas.height);
  if (points.length < 2) return;
  context.lineWidth = Math.max(3, canvas.width / 90); context.lineCap = "round"; context.lineJoin = "round";
  for (let i=0;i<points.length-1;i++) { const a=points[i],b=points[i+1],hue=240*(i/(points.length-1)); context.strokeStyle=`hsl(${hue},88%,60%)`; context.beginPath(); context.moveTo((a.x/.6+1)*canvas.width/2,(-a.y/.6+1)*canvas.height/2); context.lineTo((b.x/.6+1)*canvas.width/2,(-b.y/.6+1)*canvas.height/2); context.stroke(); }
}

async function poll() {
  if (!state.characteristic) return;
  try {
    const parsed = parseStroke(await state.characteristic.readValue()); if (!parsed) return;
    draw($("live"), parsed.points);
    if (parsed.gestureState === 2 && parsed.points.length >= 20) {
      const signature = parsed.points.map((point)=>`${point.x.toFixed(3)},${point.y.toFixed(3)}`).join(";");
      if (signature !== state.lastDone) { state.lastDone = signature; state.strokes.push({ id: crypto.randomUUID(), label: $("label").value, sessionId: sessionId(), points: parsed.points }); render(); }
    }
  } catch (error) { stopPolling(); status(`Recorder paused: ${error.message}`); }
}
function stopPolling(){ if(state.timer) clearInterval(state.timer); state.timer=null; }

async function connect() {
  if (!navigator.bluetooth) { status("Web Bluetooth is unavailable. Use desktop Chrome or Edge over HTTPS."); return; }
  try {
    status("Choose the BLESense device…"); const device=await navigator.bluetooth.requestDevice({filters:[{services:[SERVICE_UUID]}]});
    device.addEventListener("gattserverdisconnected",()=>{stopPolling();state.characteristic=null;status("Wand disconnected. Your captured data is still available for export.");});
    const server=await device.gatt.connect(); const service=await server.getPrimaryService(SERVICE_UUID); state.characteristic=await service.getCharacteristic(STROKE_UUID);
    stopPolling(); state.timer=setInterval(poll,80); status(`Connected to ${device.name || "Magic Wand"}. Stop moving briefly after each gesture.`);
  } catch(error){status(`Connection failed: ${error.message}`)}
}

function sessions() { const unique=new Map(); for(const stroke of state.strokes){const [label,split,...stamp]=stroke.sessionId.split("-"); unique.set(stroke.sessionId,{id:stroke.sessionId,label,split,stamp:stamp.join("-")});} return [...unique.values()]; }
function render(){ const counts=Object.fromEntries(labels.map(label=>[label,state.strokes.filter(stroke=>stroke.label===label).length])); $("summary").textContent=`${state.strokes.length} strokes · ${labels.map(label=>`${label} ${counts[label]}`).join(" · ")}`; const gallery=$("gallery");gallery.replaceChildren(); for(const stroke of state.strokes.slice().reverse()){const card=document.createElement("article");card.className="card";const canvas=document.createElement("canvas");canvas.width=180;canvas.height=180;draw(canvas,stroke.points);const footer=document.createElement("footer");const name=document.createElement("strong");name.textContent=stroke.label;const remove=document.createElement("button");remove.textContent="Delete";remove.onclick=()=>{state.strokes=state.strokes.filter(item=>item.id!==stroke.id);render()};footer.append(name,remove);card.append(canvas,footer);gallery.append(card);}}
function exportData(){if(!state.strokes.length){status("Capture or import at least one stroke before exporting.");return}const payload={schemaVersion:1,exportedAt:new Date().toISOString(),boardProfile:$("board").value,sessions:sessions().map(({stamp,...session})=>session),strokes:state.strokes};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));const link=document.createElement("a");link.href=url;link.download=`magic-wand-${new Date().toISOString().slice(0,10)}.json`;link.click();URL.revokeObjectURL(url);status(`Exported ${state.strokes.length} strokes.`)}
async function importData(event){try{const data=JSON.parse(await event.target.files[0].text());if(data.schemaVersion!==1||!Array.isArray(data.strokes))throw new Error("This is not a schema v1 Magic Wand dataset");state.strokes=data.strokes;render();status(`Imported ${state.strokes.length} strokes. Review session and split metadata before training.`)}catch(error){status(`Import failed: ${error.message}`)}}
$("connect").onclick=connect;$("export").onclick=exportData;$("import").onchange=importData;render();
