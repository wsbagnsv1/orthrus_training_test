#!/usr/bin/env python
"""
Standalone WebUI for Orthrus training metrics.

Usage:
    python webui.py                          # serves metrics.jsonl from ./checkpoints
    python webui.py --file path/to/metrics.jsonl --port 8080
    python webui.py --file path/to/logs/     # auto-finds metrics.jsonl in dir

Open http://localhost:8080 in browser.
"""

import argparse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_metrics_path = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            data = []
            if _metrics_path and os.path.exists(_metrics_path):
                with open(_metrics_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML.encode())

    def log_message(self, format, *args):
        pass


_HTML = r"""<!DOCTYPE html>
<html><head><meta charset=utf-8><title>Orthrus Training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2"></script>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--green:#3fb950;--cyan:#58a6ff;
  --yellow:#d2991d;--magenta:#bc8cff;--red:#f85149;--text:#c9d1d9;--muted:#8b949e}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--text);padding:16px 24px;min-height:100vh}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
h1{font-size:20px;font-weight:600;letter-spacing:-0.3px}
.controls{display:flex;gap:8px;align-items:center}
.controls select,.controls button{background:var(--card);color:var(--text);
  border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer}
.controls button:hover{background:var(--border)}
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:14px 16px;position:relative;overflow:hidden}
.stat-card .label{font-size:12px;color:var(--muted);margin-bottom:4px}
.stat-card .value{font-size:28px;font-weight:700;letter-spacing:-0.5px}
.stat-card .delta{font-size:12px;margin-top:2px}
.stat-card canvas{position:absolute;right:8px;top:50%;transform:translateY(-50%);opacity:0.3;pointer-events:none}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}
.chart-card h2{font-size:14px;color:var(--muted);margin-bottom:12px}
.chart-card .tabs{display:flex;gap:0;margin-bottom:8px}
.chart-card .tabs button{background:transparent;color:var(--muted);border:none;
  padding:6px 14px;font-size:12px;cursor:pointer;border-bottom:2px solid transparent}
.chart-card .tabs button.active{color:var(--text);border-bottom-color:var(--green)}
.chart-wrap{position:relative;height:320px}
.chart-wrap canvas{width:100%!important;height:100%!important}
.delta-up{color:var(--green)}.delta-down{color:var(--red)}.delta-flat{color:var(--muted)}
#status{font-size:12px;color:var(--muted)}
</style></head><body>
<header>
  <h1>Orthrus SmolLM2 Training</h1>
  <div class=controls>
    <button onclick="resetView()">↺ Reset</button>
    <button onclick="resetZoom()">⊞ Zoom Reset</button>
    <button onclick="poll()">↻ Refresh</button>
    <label style="font-size:12px;color:var(--muted);margin-left:4px">From</label>
    <input id=fromStep type=number value=0 min=0 step=100 style="background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:12px;width:80px" onchange="applyFilter()">
    <select id=poll onchange="startPoll(this.value)">
      <option value=1 selected>1s</option><option value=2>2s</option>
      <option value=5>5s</option><option value=15>15s</option><option value=60>60s</option>
    </select>
    <span id=status></span>
  </div>
</header>

<div class=stats-grid>
  <div class=stat-card>
    <div class=label>KL Loss</div>
    <div class=value id=kl-val style=color:var(--green)>—</div>
    <div class=delta id=kl-delta></div>
    <canvas id=kl-spark width=120 height=40></canvas>
  </div>
  <div class=stat-card>
    <div class=label>Validation KL</div>
    <div class=value id=vkl-val style=color:var(--cyan)>—</div>
    <div class=delta id=vkl-delta></div>
    <canvas id=vkl-spark width=120 height=40></canvas>
  </div>
  <div class=stat-card>
    <div class=label>Acceptance Rate</div>
    <div class=value id=acc-val style=color:var(--yellow)>—</div>
    <div class=delta id=acc-delta></div>
    <canvas id=acc-spark width=120 height=40></canvas>
  </div>
  <div class=stat-card>
    <div class=label>Gradient Norm</div>
    <div class=value id=gn-val style=color:var(--magenta)>—</div>
    <div class=delta id=gn-delta></div>
    <canvas id=gn-spark width=120 height=40></canvas>
  </div>
</div>

<div class=chart-card>
  <div class=tabs>
    <button class=active onclick="switchChart('loss')">KL Loss</button>
    <button onclick="switchChart('val')">Val KL</button>
    <button onclick="switchChart('acc')">Acceptance %</button>
    <button onclick="switchChart('gn')">Grad Norm</button>
    <button onclick="switchChart('all')">All</button>
  </div>
  <div class=chart-wrap><canvas id=main></canvas></div>
</div>

<div class=chart-card>
  <h2>Δ Val KL (descent rate per eval)</h2>
  <div class=chart-wrap><canvas id=logChart></canvas></div>
</div>

<script>
const C={loss:'#3fb950',val:'#58a6ff',acc:'#d2991d',gn:'#bc8cff'};
const zoomPlug={zoom:{drag:{enabled:true},mode:'x'},pan:{enabled:true,mode:'x'}};

let mainChart=new Chart(document.getElementById('main'),{type:'line',
  data:{datasets:[
    {label:'KL Loss',data:[],borderColor:C.loss,pointRadius:0,tension:0.2,yAxisID:'y'},
    {label:'Val KL',data:[],borderColor:C.val,pointRadius:0,hidden:true,yAxisID:'y'},
    {label:'Accept %',data:[],borderColor:C.acc,pointRadius:0,hidden:true,yAxisID:'y1'},
    {label:'Grad Norm',data:[],borderColor:C.gn,pointRadius:0,hidden:true,yAxisID:'y1'},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:200},
    scales:{
      x:{type:'linear',ticks:{color:'#8b949e',maxTicksLimit:15},grid:{color:'#21262d'}},
      y:{type:'linear',position:'left',ticks:{color:'#8b949e'},grid:{color:'#21262d'}},
      y1:{type:'linear',position:'right',ticks:{color:'#8b949e'},grid:{drawOnChartArea:false}}},
    plugins:{zoom:zoomPlug,legend:{display:false}}}});

let logChart=new Chart(document.getElementById('logChart'),{type:'bar',
  data:{datasets:[{label:'Δ Val KL',data:[],backgroundColor:pts=>{
    let v=pts.raw?.y||0;return v>0?'rgba(63,185,80,0.6)':'rgba(248,81,73,0.6)'},
    borderColor:pts=>{let v=pts.raw?.y||0;return v>0?'#3fb950':'#f85149'},barThickness:8}]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:200},
    scales:{
      x:{type:'linear',ticks:{color:'#8b949e',maxTicksLimit:15},grid:{color:'#21262d'}},
      y:{type:'linear',ticks:{color:'#8b949e'},grid:{color:'#21262d'}}},
    plugins:{zoom:zoomPlug,legend:{display:false}}}});

let allPoints=[],lastStep=-1,pollTimer,prevVals={},fromStep=0;
function applyFilter(){
  fromStep=parseInt(document.getElementById('fromStep').value)||0;
  renderAll(allPoints);
}
let sparkCharts={};

function makeSpark(id,color){
  return new Chart(id,{type:'line',data:{datasets:[{data:[],borderColor:color,
    pointRadius:0,borderWidth:1}]},
    options:{animation:false,maintainAspectRatio:true,
      scales:{x:{display:false},y:{display:false}},
      plugins:{legend:{display:false}}}})}
sparkCharts.loss=makeSpark('kl-spark',C.loss);
sparkCharts.val=makeSpark('vkl-spark',C.val);
sparkCharts.acc=makeSpark('acc-spark',C.acc);
sparkCharts.gn=makeSpark('gn-spark',C.gn);

function switchChart(mode){
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  let ds=mainChart.data.datasets;
  let showAll=mode==='all';
  ds[0].hidden=!(showAll||mode==='loss');ds[0].label='KL Loss';
  ds[1].hidden=!(showAll||mode==='val');ds[1].label='Val KL';
  ds[2].hidden=!(showAll||mode==='acc');ds[2].label='Accept %';
  ds[3].hidden=!(showAll||mode==='gn');ds[3].label='Grad Norm';
  if(showAll)mainChart.options.plugins.legend.display=true;
  else mainChart.options.plugins.legend.display=false;
  mainChart.update();
}

function deltaClass(cur,prev,lowerIsBetter){
  if(!prev||prev===0)return'delta-flat';
  let pct=(cur-prev)/Math.abs(prev||1);
  if(lowerIsBetter)pct=-pct;
  if(pct<-0.01)return'delta-down';if(pct>0.01)return'delta-up';return'delta-flat'}
function deltaSign(cur,prev,lowerIsBetter){
  if(!prev||prev===0)return'';let pct=((cur-prev)/Math.abs(prev||1))*100;
  if(lowerIsBetter)pct=-pct;return pct>0?'+'+pct.toFixed(1)+'%':pct.toFixed(1)+'%'}
function updateStat(valId,deltaId,val,fmt,lowerIsBetter){
  let el=document.getElementById(valId);
  let prev=prevVals[valId];el.textContent=typeof fmt==='function'?fmt(val):(val!=null?val.toFixed(2):'—');
  let d=document.getElementById(deltaId);d.textContent=deltaSign(val,prev,lowerIsBetter);
  d.className='delta '+deltaClass(val,prev,lowerIsBetter);prevVals[valId]=val}

function loadAll(){
  fetch('/data').then(r=>r.json()).then(d=>{
    allPoints=d;renderAll(d);
    document.getElementById('status').textContent=d.length+' pts · step '+(d[d.length-1]?.step||'?');
    lastStep=d[d.length-1]?.step||-1;
  }).catch(()=>document.getElementById('status').textContent='⚡ waiting...')}

function poll(){
  document.getElementById('status').textContent='↻ loading...';
  fetch('/data').then(r=>r.json()).then(d=>{
    if(!d.length)return;
    let fresh=d.filter(p=>p.step>lastStep);
    if(!fresh.length){document.getElementById('status').textContent=allPoints.length+' pts · step '+lastStep;return;}
    allPoints=allPoints.concat(fresh);
    if(allPoints.length>20000)allPoints=allPoints.slice(-15000);
    lastStep=d[d.length-1].step;
    renderAll(allPoints);
    document.getElementById('status').textContent=allPoints.length+' pts · step '+lastStep;
  }).catch(e=>{document.getElementById('status').textContent='⚠ '+e.message})}
function renderAll(pts){
  pts=pts.filter(p=>p.step>=fromStep);
  // KL Loss: only training steps (val_kl==null), NOT eval entries that also carry loss
  let kl=pts.filter(p=>p.loss!=null&&p.val_kl==null),
      vk=pts.filter(p=>p.val_kl!=null),
      ac=pts.filter(p=>p.accept_rate!=null),gn=pts.filter(p=>p.grad_norm!=null);
  mainChart.data.datasets[0].data=kl.map(p=>({x:p.step,y:p.loss}));
  mainChart.data.datasets[1].data=vk.map(p=>({x:p.step,y:p.val_kl}));
  mainChart.data.datasets[2].data=ac.map(p=>({x:p.step,y:p.accept_rate*100}));
  mainChart.data.datasets[3].data=gn.map(p=>({x:p.step,y:p.grad_norm}));
  sparkCharts.loss.data.datasets[0].data=kl.slice(-200).map(p=>p.loss);
  sparkCharts.val.data.datasets[0].data=vk.slice(-200).map(p=>p.val_kl);
  sparkCharts.acc.data.datasets[0].data=ac.slice(-200).map(p=>p.accept_rate*100);
  sparkCharts.gn.data.datasets[0].data=gn.slice(-200).map(p=>p.grad_norm);
  sparkCharts.loss.update();sparkCharts.val.update();sparkCharts.acc.update();sparkCharts.gn.update();
  mainChart.update();
  logChart.data.datasets[0].data=vk.length>1?vk.slice(1).map((p,i)=>({x:p.step,y:-(p.val_kl-vk[i].val_kl)})):[];
  logChart.update();
  let s=pts[pts.length-1]||{};
  updateStat('kl-val','kl-delta',s.loss,null,true);
  updateStat('vkl-val','vkl-delta',s.val_kl,null,true);
  updateStat('acc-val','acc-delta',s.accept_rate?s.accept_rate*100:null,v=>v!=null?v.toFixed(1)+'%':'—',false);
  updateStat('gn-val','gn-delta',s.grad_norm,null,true);
}

function startPoll(sec){
  clearInterval(pollTimer);
  pollTimer=setInterval(poll,sec*1000);
}
// Poll on tab focus for instant refresh after being backgrounded
document.addEventListener('visibilitychange',function(){
  if(!document.hidden){poll();startPoll(document.getElementById('poll').value)}
});
function resetView(){lastStep=-1;allPoints=[];
  fromStep=0;document.getElementById('fromStep').value=0;
  mainChart.data.datasets.forEach(d=>d.data=[]);
  logChart.data.datasets[0].data=[];
  Object.values(sparkCharts).forEach(s=>s.data.datasets[0].data=[]);
  mainChart.update();logChart.update();Object.values(sparkCharts).forEach(s=>s.update());
  prevVals={};['kl-val','vkl-val','acc-val','gn-val'].forEach(id=>{
    document.getElementById(id+'-val').textContent='—';document.getElementById(id+'-delta').textContent=''})}
function resetZoom(){mainChart.resetZoom()}
loadAll();startPoll(1);
</script></body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orthrus Training WebUI")
    parser.add_argument("--file", type=str, default="checkpoints/metrics.jsonl",
                        help="Path to metrics JSONL file or directory containing it")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    path = Path(args.file)
    if path.is_dir():
        path = path / "metrics.jsonl"
    _metrics_path = str(path.resolve())

    if not os.path.exists(_metrics_path):
        print(f"⚠ Metrics file not found: {_metrics_path}")
        print(f"  UI will start but show no data until training logs metrics.")
    else:
        print(f"✓ Reading metrics from: {_metrics_path}")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"✓ WebUI → http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Shutdown")
