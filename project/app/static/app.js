const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (v, d = 1) => (100 * v).toFixed(d) + '%';
const charts = {};
let R = null;

// ---------- tabs ----------
document.querySelectorAll('#tabs button').forEach((b) => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === b.dataset.tab));
  history.replaceState(null, '', '#' + b.dataset.tab);
  Object.values(charts).forEach((c) => c.resize());
  if (b.dataset.tab === 'qec' && !charts.qec) runQEC();
  if (b.dataset.tab === 'route' && !charts.qaoa) runRoute();
});

function seg(id, onChange) {
  const el = $(id);
  el.querySelectorAll('button').forEach((b) => b.onclick = () => {
    el.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); onChange && onChange();
  });
  return () => +el.querySelector('button.on').dataset.v;
}

function chart(key, el, cfg) {
  if (charts[key]) charts[key].destroy();
  Chart.defaults.color = css('--muted'); Chart.defaults.borderColor = css('--line');
  Chart.defaults.font.family = 'Inter';
  charts[key] = new Chart(el, cfg);
}

const api = (url, body) => fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body) } : {}).then((r) => r.json());

// ---------- overview KPIs ----------
async function loadResults() {
  R = await api('/api/results');
  const nK = Math.max(...R.comparison.map((r) => r.n).filter((n) => n <= 1000));
  const comp = R.comparison.filter((r) => r.n === nK);
  const mean = (name) => { const a = comp.filter((r) => r.model === name).map((r) => r.acc); return a.reduce((s, v) => s + v, 0) / a.length; };
  const classical = ['Logistic regression', 'SVM (RBF)', 'Random forest', 'MLP (16-16)'];
  const bestC = classical.map((n) => [n, mean(n)]).sort((a, b) => b[1] - a[1])[0];
  const q = R.qec.find((r) => r.p === 0.01);
  $('kpis').innerHTML = [
    ['q', pct(mean('Re-uploading VQC (ours)')), `Re-uploading VQC accuracy (n=${nK}, 2017–18 test)`],
    ['c', pct(bestC[1]), `Best classical: ${bestC[0]}`],
    ['q', R.latency['VQC – exact simulator'].median_ms.toFixed(1) + ' ms', 'VQC inference per snapshot (budget 1000 ms)'],
    ['q', (q.unprotected / Math.max(q.shor, 1e-6)).toFixed(0) + '×', 'Fewer logical errors with Shor code at p = 1%'],
  ].map(([c, v, l]) => `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
  renderResults();
}

// ---------- prediction ----------
const dayType = seg('dayType', predict);
const WEATHER = ['Clear', 'Clouds', 'Mist/fog', 'Rain', 'Snow/storm'];
let tPred;
['hour', 'temp', 'weather'].forEach((id) => $(id).oninput = () => { clearTimeout(tPred); tPred = setTimeout(predict, 120); });

async function predict() {
  $('hourOut').textContent = String($('hour').value).padStart(2, '0') + ':00';
  $('tempOut').textContent = $('temp').value + ' °C';
  const r = await api('/api/predict', { hour: +$('hour').value, day_type: dayType(), temp_c: +$('temp').value, weather: +$('weather').value });
  const d = $('decision'); d.textContent = r.decision; d.className = 'decision ' + (r.decision === 'Congested' ? 'yes' : 'no');
  $('action').textContent = '→ ' + r.action;
  $('angles').innerHTML = 'x = [' + r.angles.map((a) => a.toFixed(3)).join(', ') + '] rad &nbsp; ⟨Z₀⟩ = ' + r.expval_z0.toFixed(3);
  $('modelTable').innerHTML = '<tr><th>Model</th><th></th><th>P(congested)</th><th></th><th>ms</th></tr>' + r.predictions.map((p, i) => `
    <tr class="${i === 0 ? 'hl' : ''}"><td>${p.model}</td><td><span class="tag ${p.family[0]}">${p.family}</span></td>
    <td>${pct(p.p)}</td><td><div class="bar"><i style="width:${100 * p.p}%;background:${p.family === 'quantum' ? css('--q') : css('--c')}"></i></div></td>
    <td>${p.ms.toFixed(2)}</td></tr>`).join('');
  $('timing').textContent = `End-to-end (preprocessing + 5 models + decision): ${r.total_ms.toFixed(1)} ms of the 1000 ms budget.`;
  const labels = r.state_probs.map((_, i) => '|' + i.toString(2).padStart(4, '0') + '⟩');
  chart('state', $('stateChart'), { type: 'bar', data: { labels, datasets: [{ data: r.state_probs,
    backgroundColor: labels.map((_, i) => ((i >> 3) & 1) ? css('--bad') : css('--q')) }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'probability' } } } } });
}

// ---------- routing ----------
const rP = seg('rP');
let rSeed = 0;
for (let i = 0; i < 8; i++) $('rEmerg').innerHTML += `<option value="${i}">Stream ${i + 1}</option>`;
$('rHour').oninput = () => $('rHourOut').textContent = String($('rHour').value).padStart(2, '0') + ':00';
$('rNew').onclick = () => { rSeed++; runRoute(); };
$('rRun').onclick = runRoute;

async function runRoute() {
  $('rRun').textContent = 'Optimising…';
  const r = await api('/api/route', { hour: +$('rHour').value, emergency: +$('rEmerg').value, p: rP(), seed: rSeed });
  $('rRun').textContent = 'Optimise routes';
  const q = r.plans.QAOA, e = r.plans.Exact, max = Math.max(...r.demands);
  $('plan').innerHTML = '<div class="streams">' + r.demands.map((d, i) => `
    <div class="stream"><span>Stream ${i + 1}${i === r.emergency ? ' 🚑' : ''}</span>
      <div class="bar"><i style="width:${100 * d / max}%;background:${q.x[i] ? css('--q') : css('--c')}"></i></div>
      <span class="pill ${q.x[i] ? 'alt' : 'main'}">${q.x[i] ? 'ALTERNATE' : 'main'} · ${d}</span></div>`).join('') + '</div>' +
    `<table class="models" style="margin-top:14px"><tr><th>Solver</th><th>cost J</th><th>alt. share</th><th>ms</th></tr>` +
    Object.entries(r.plans).map(([k, v]) => `<tr class="${k === 'QAOA' ? 'hl' : ''}"><td>${k}</td><td>${v.cost.toFixed(4)}</td>
      <td>${pct(v.alt_share, 0)}</td><td>${v.ms.toFixed(1)}</td></tr>`).join('') + '</table>' +
    `<p class="muted">${Math.abs(q.cost - e.cost) < 1e-9 ? '✓ QAOA found the optimal plan.' : `QAOA plan is ${pct((q.cost - e.cost) / e.cost, 2)} above optimum.`}
     Classical feasibility check: emergency stream ${q.x[r.emergency] ? '✗ diverted — rejected' : '✓ stays on main road'}.</p>`;
  chart('qaoa', $('qaoaChart'), { type: 'bar', data: { labels: r.top_states.map((s) => s.bits),
    datasets: [{ data: r.top_states.map((s) => s.prob), backgroundColor: r.top_states.map((s) =>
      Math.abs(s.cost - e.cost) < 1e-9 ? css('--ok') : css('--q')) }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'probability' } } } } });
  $('qaoaInfo').textContent = `Green = optimal plan. Approximation ratio ${r.approx_ratio.toFixed(3)}; P(optimal) per shot ${pct(r.p_opt, 2)} (random guessing: 0.39%).`;
}

// ---------- QEC ----------
let tQ;
$('pErr').oninput = () => { $('pOut').textContent = ($('pErr').value / 10).toFixed(1) + '%'; clearTimeout(tQ); tQ = setTimeout(runQEC, 200); };
async function runQEC() {
  const p = $('pErr').value / 1000;
  const r = await api('/api/qec?p=' + p);
  const names = { unprotected: 'Unprotected', bit_flip: 'Bit-flip (3)', phase_flip: 'Phase-flip (3)', shor: 'Shor (9)' };
  const protects = { unprotected: '—', bit_flip: 'X only', phase_flip: 'Z only', shor: 'X, Y, Z' };
  chart('qec', $('qecChart'), { type: 'bar', data: { labels: Object.values(names),
    datasets: [{ label: 'logical error', data: Object.keys(names).map((k) => r[k].p_logical),
      backgroundColor: Object.keys(names).map((k) => r[k].p_logical < r.unprotected.p_logical ? css('--ok') : k === 'unprotected' ? css('--muted') : css('--bad')) }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'P(logical error)' } } } } });
  $('qecTable').innerHTML = '<tr><th>Code</th><th>qubits</th><th>protects</th><th>p_L</th></tr>' + Object.keys(names).map((k) =>
    `<tr class="${k === 'shor' ? 'hl' : ''}"><td>${names[k]}</td><td>${r[k].qubits}</td><td>${protects[k]}</td><td>${pct(r[k].p_logical, 2)}</td></tr>`).join('');
}

// ---------- results ----------
function renderResults() {
  const models = ['Re-uploading VQC (ours)', 'Standard VQC', 'Quantum-kernel SVM', 'Logistic regression', 'SVM (RBF)', 'Random forest', 'MLP (16-16)'];
  const colors = ['#7c3aed', '#c4b5fd', '#a78bfa', '#94a3b8', '#38bdf8', '#0284c7', '#475569'];
  const sizes = [...new Set(R.comparison.map((r) => r.n))];
  chart('lc', $('lcChart'), { type: 'line', data: { labels: sizes, datasets: models.map((m, i) => ({ label: m,
    data: sizes.map((n) => { const a = R.comparison.filter((r) => r.model === m && r.n === n).map((r) => r.acc); return a.reduce((s, v) => s + v, 0) / a.length; }),
    borderColor: colors[i], backgroundColor: colors[i], borderWidth: i === 0 ? 3 : 1.5, borderDash: i > 0 && i < 3 ? [4, 3] : [] })) },
    options: { plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: { x: { title: { display: true, text: 'training snapshots' } }, y: { title: { display: true, text: 'test accuracy' } } } } });

  const groups = {};
  R.ablation.forEach((r) => (groups[r.variant] ??= []).push(r));
  $('ablTable').innerHTML = '<tr><th>Variant</th><th>params</th><th>accuracy</th><th>AUC</th></tr>' + Object.entries(groups).map(([v, rs]) => {
    const m = (k) => rs.reduce((s, r) => s + r[k], 0) / rs.length;
    return `<tr class="${v.includes('ours') ? 'hl' : ''}"><td>${v.trim()}</td><td>${rs[0].n_params}</td><td>${pct(m('acc'))}</td><td>${m('auc').toFixed(3)}</td></tr>`;
  }).join('');

  const qg = {};
  R.qaoa.forEach((r) => (qg[r.method] ??= []).push(r));
  const f = (rs, k, fn) => rs[0][k] === undefined ? '—' : fn(rs.reduce((s, r) => s + r[k], 0) / rs.length);
  $('qaoaTable').innerHTML = '<tr><th>Method</th><th>⟨r⟩</th><th>gap of best plan</th><th>ms</th></tr>' + Object.entries(qg).map(([m, rs]) =>
    `<tr class="${m.startsWith('QAOA') ? 'hl' : ''}"><td>${m}</td><td>${f(rs, 'approx_ratio', (v) => v.toFixed(3))}</td>
     <td>${f(rs, 'gap', (v) => pct(v, 2))}</td><td>${f(rs, 'ms', (v) => v.toFixed(1))}</td></tr>`).join('');
}

loadResults().then(predict).then(() => {
  const t = document.querySelector(`#tabs button[data-tab="${location.hash.slice(1)}"]`);
  if (t) t.click();
});
