"""Embedded browser UI for entering prompts from another machine."""

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGI Web Agent</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; font-family:system-ui,-apple-system,Segoe UI,sans-serif; background:#0b1020; color:#e8edf7; }
main { max-width:1100px; margin:0 auto; padding:28px; }
header { display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:20px; }
h1 { margin:0; font-size:26px; }
.badge { padding:6px 10px; border-radius:999px; background:#18233d; font-size:13px; }
.card { background:#111a2e; border:1px solid #263451; border-radius:14px; padding:18px; margin-bottom:16px; }
label { display:block; margin-bottom:9px; font-weight:600; }
textarea { width:100%; min-height:130px; resize:vertical; border:1px solid #334463; border-radius:10px; padding:13px; background:#0a1121; color:#fff; font:inherit; }
.controls { display:flex; gap:10px; margin-top:12px; }
button { border:0; border-radius:9px; padding:10px 18px; background:#4f7cff; color:white; font-weight:700; cursor:pointer; }
button.secondary { background:#253452; }
button:disabled { opacity:.5; cursor:not-allowed; }
pre { white-space:pre-wrap; overflow:auto; background:#080d18; border-radius:10px; padding:14px; min-height:120px; }
#status { color:#9eb3d8; }
small { color:#91a0bb; }
</style>
</head>
<body>
<main>
<header><h1>AGI Web Agent</h1><span id="health" class="badge">checking...</span></header>
<section class="card">
<label for="prompt">Prompt</label>
<textarea id="prompt" autofocus placeholder="Example: برو به https://example.com و صفحه را بررسی کن
Example: search for Docker architecture"></textarea>
<div class="controls">
<button id="run">Execute</button>
<button id="clear" class="secondary">Clear</button>
</div>
<p id="status">Ready.</p>
</section>
<section class="card"><h2>Agent response</h2><pre id="response">No task executed yet.</pre></section>
<section class="card"><h2>Current browser observation</h2><pre id="observation">Loading...</pre></section>
<section class="card"><h2>Execution trace</h2><pre id="trace">No actions yet.</pre></section>
</main>
<script>
const $ = id => document.getElementById(id);
async function request(path, options={}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
async function refresh() {
  try {
    const data = await request('/api/observe');
    $('observation').textContent = JSON.stringify(data.observation, null, 2);
    $('health').textContent = 'online';
  } catch (error) {
    $('health').textContent = 'offline';
    $('status').textContent = error.message;
  }
}
$('run').onclick = async () => {
  const prompt = $('prompt').value.trim();
  if (!prompt) return;
  $('run').disabled = true;
  $('status').textContent = 'Agent is working...';
  try {
    const data = await request('/api/prompt', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt})
    });
    $('response').textContent = data.response || 'Task completed.';
    $('observation').textContent = JSON.stringify(data.observation, null, 2);
    $('trace').textContent = JSON.stringify(data.steps || [], null, 2);
    $('status').textContent = `Completed in ${data.steps ? data.steps.length : 0} action(s).`;
  } catch (error) {
    $('status').textContent = `Error: ${error.message}`;
  } finally { $('run').disabled = false; }
};
$('clear').onclick = () => { $('prompt').value=''; $('response').textContent='No task executed yet.'; $('trace').textContent='No actions yet.'; };
refresh();
</script>
</body>
</html>'''
