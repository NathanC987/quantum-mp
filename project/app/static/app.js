const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (v, d = 1) => (100 * v).toFixed(d) + '%';
const num = (v) => Math.round(v).toLocaleString('en-US');
const hh = (h) => String(h).padStart(2, '0') + ':00';
const charts = {};
let R = null, O = null, THR = 3340;

const busyPill = (b) => `<span class="pill ${b ? 'busy' : 'quiet'}">${b ? 'Busy' : 'Not busy'}</span>`;
const mark = (ok) => `<span class="mark ${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✗'}</span>`;

// ---------- tabs ----------
function openTab(name) {
  const b = document.querySelector(`#tabs button[data-tab="${name}"]`);
  if (!b) return;
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === name));
  history.replaceState(null, '', '#' + name);
  window.scrollTo(0, 0);
  Object.values(charts).forEach((c) => c.resize());
  if (name === 'qec' && !charts.qec) runQEC();
  if (name === 'route' && !charts.qaoa) runRoute();
  if (name === 'check' && !charts.day) loadDay();
}
document.querySelectorAll('#tabs button').forEach((b) => b.onclick = () => openTab(b.dataset.tab));
document.querySelectorAll('.guide a').forEach((a) => a.onclick = (e) => { e.preventDefault(); openTab(a.hash.slice(1)); });

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

// ---------- 1. the problem ----------
async function loadOverview() {
  [O, R] = await Promise.all([api('/api/overview'), api('/api/results')]);
  THR = O.threshold;
  $('nRows').textContent = num(O.n_rows);
  $('span').textContent = `${O.first} – ${O.last}`;
  document.querySelectorAll('#nTest, .nTest').forEach((e) => e.textContent = num(O.n_test));
  document.querySelectorAll('.thr').forEach((e) => e.textContent = num(THR));

  chart('hour', $('hourChart'), { type: 'line', data: { labels: [...Array(24).keys()].map(hh), datasets: [
    { label: 'Weekday', data: O.weekday, borderColor: css('--c'), backgroundColor: css('--c'), tension: 0.3, borderWidth: 2.5, pointRadius: 0 },
    { label: 'Weekend / holiday', data: O.weekend, borderColor: css('--q'), backgroundColor: css('--q'), tension: 0.3, borderWidth: 2.5, pointRadius: 0 },
    { label: `Busy line (${num(THR)} cars)`, data: Array(24).fill(THR), borderColor: css('--bad'), borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
  ] }, options: { interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { boxWidth: 12 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${num(c.raw)} cars` } } },
    scales: { y: { title: { display: true, text: 'average cars per hour' } }, x: { ticks: { maxTicksLimit: 12 } } } } });

  $('sampleTable').innerHTML = '<tr><th>When</th><th>Weather</th><th>Temp.</th><th>Weekend / holiday</th><th>Cars counted</th><th>Correct answer</th></tr>' +
    O.sample.map((r) => `<tr><td>${r.when}</td><td>${r.weather}</td><td>${r.temp_c} °C</td><td>${r.holiday ? 'yes' : 'no'}</td>
      <td class="num">${num(r.cars)}</td><td>${busyPill(r.busy)}</td></tr>`).join('');

  const s = Object.fromEntries(O.scores.map((x) => [x.key, x]));
  const total = O.n_test;
  const bestC = O.scores.filter((x) => x.family === 'classical').sort((a, b) => b.acc - a.acc)[0];
  const q = R.qec.find((r) => r.p === 0.01);
  $('kpis').innerHTML = [
    ['q', pct(s.vqc.acc), `The quantum model was right in <b>${num(s.vqc.tp + s.vqc.tn)} of ${num(total)}</b> test hours it had never seen`],
    ['c', pct(bestC.acc), `Best ordinary (classical) model, ${bestC.model}, on the same hours. The two are on par`],
    ['q', R.latency['VQC – exact simulator'].median_ms.toFixed(1) + ' ms', 'Time for one quantum prediction (the limit is 1 second = 1000 ms)'],
    ['q', (q.unprotected / Math.max(q.shor, 1e-6)).toFixed(0) + '×', 'Fewer read-out errors when the Shor error-correcting code protects a qubit (at 1% noise)'],
  ].map(([c, v, l]) => `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');

  // 2. scorecard
  $('scoreTable').innerHTML = '<tr><th>Model</th><th>Type</th><th>Right answers</th><th>Accuracy</th><th></th></tr>' +
    O.scores.map((x) => `<tr class="${x.key === 'vqc' ? 'hl' : ''}"><td>${x.model}${x.key === 'vqc' ? ' (ours)' : ''}</td>
      <td><span class="tag ${x.family[0]}">${x.family}</span></td><td class="num">${num(x.tp + x.tn)} / ${num(total)}</td>
      <td class="num">${pct(x.acc)}</td><td><div class="bar"><i style="width:${100 * x.acc}%;background:${x.family === 'quantum' ? css('--q') : css('--c')}"></i></div></td></tr>`).join('');
  const v = s.vqc;
  $('confusion').innerHTML = `
    <div></div><div class="ch">Really busy</div><div class="ch">Really not busy</div>
    <div class="rh">Model said busy</div><div class="cell ok"><b>${num(v.tp)}</b>correct</div><div class="cell bad"><b>${num(v.fp)}</b>false alarm</div>
    <div class="rh">Model said not busy</div><div class="cell bad"><b>${num(v.fn)}</b>missed a busy hour</div><div class="cell ok"><b>${num(v.tn)}</b>correct</div>`;
  $('dayPick').min = O.test_first; $('dayPick').max = O.test_last;
  renderResults();
}

// ---------- 2. one real day ----------
let daySeed = 1;
$('dayPick').onchange = () => loadDay({ date: $('dayPick').value });
$('dayRandom').onclick = () => loadDay({ seed: daySeed++ });
$('dayMistake').onclick = () => loadDay({ seed: daySeed++, mistakes: true });

async function loadDay(q = { date: '2017-05-09' }) {
  const d = await api('/api/day?' + new URLSearchParams(q));
  $('dayPick').value = d.date;
  const rows = d.rows;
  const ok = (r, k) => (r.p[k] > 0.5) === (r.actual === 1);
  const nQ = rows.filter((r) => ok(r, 'vqc')).length, nC = rows.filter((r) => ok(r, 'mlp')).length;
  $('dayTitle').textContent = d.label;
  $('daySub').textContent = d.holiday ? 'Weekend or public holiday' : 'Working weekday';
  $('daySummary').innerHTML = `<div><b class="q">${nQ} / ${rows.length}</b> hours right: quantum model</div>
    <div><b class="c">${nC} / ${rows.length}</b> hours right: classical MLP</div>
    <div><b>${rows.filter((r) => r.actual).length}</b> hours were really busy</div>`;

  chart('day', $('dayChart'), { data: { labels: rows.map((r) => r.time), datasets: [
    { type: 'bar', label: 'Cars really counted', data: rows.map((r) => r.cars), borderRadius: 3,
      backgroundColor: rows.map((r) => ok(r, 'vqc') ? css('--ok') : css('--bad')) },
    { type: 'line', label: 'Busy line', data: rows.map(() => THR), borderColor: css('--text'), borderDash: [6, 4], borderWidth: 1.2, pointRadius: 0 },
  ] }, options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: {
      label: (c) => c.datasetIndex ? `Busy line: ${num(THR)}` : `Real: ${num(c.raw)} cars (${rows[c.dataIndex].actual ? 'busy' : 'not busy'})`,
      afterLabel: (c) => c.datasetIndex ? '' : `Quantum model: ${pct(rows[c.dataIndex].p.vqc, 0)} chance of busy ${ok(rows[c.dataIndex], 'vqc') ? '✓' : '✗'}` } } },
    scales: { y: { title: { display: true, text: 'cars that hour' } } } } });

  $('dayTable').innerHTML = `<tr><th>Time</th><th>Weather</th><th>Temp.</th><th>Cars really counted</th><th>Correct answer</th>
    <th class="qh">Quantum model says</th><th class="qh"></th><th>Classical MLP says</th><th></th></tr>` + rows.map((r) => {
    const pq = r.p.vqc > 0.5, pc = r.p.mlp > 0.5;
    return `<tr class="${ok(r, 'vqc') ? '' : 'wrong'}"><td>${r.time}</td><td>${r.weather}</td><td>${r.temp_c} °C</td>
      <td class="num">${num(r.cars)}</td><td>${busyPill(r.actual)}</td>
      <td>${busyPill(pq)} <span class="muted">${pct(r.p.vqc, 0)}</span></td><td>${mark(ok(r, 'vqc'))}</td>
      <td>${busyPill(pc)} <span class="muted">${pct(r.p.mlp, 0)}</span></td><td>${mark(ok(r, 'mlp'))}</td></tr>`;
  }).join('');
}

// ---------- 3. try it yourself ----------
const dayType = seg('dayType', predict);
let tPred;
['hour', 'temp', 'weather'].forEach((id) => $(id).oninput = () => { clearTimeout(tPred); tPred = setTimeout(predict, 120); });

async function predict() {
  $('hourOut').textContent = hh($('hour').value);
  $('tempOut').textContent = $('temp').value + ' °C';
  const r = await api('/api/predict', { hour: +$('hour').value, day_type: dayType(), temp_c: +$('temp').value, weather: +$('weather').value });
  const busy = r.decision === 'Busy';
  const d = $('decision'); d.innerHTML = `${r.decision} <small>${pct(r.predictions[0].p, 0)} chance of more than ${num(r.threshold)} cars</small>`;
  d.className = 'decision ' + (busy ? 'yes' : 'no');
  $('action').textContent = 'What the city would do: ' + r.action.toLowerCase();
  $('angles').innerHTML = 'clues as angles x = [' + r.angles.map((a) => a.toFixed(3)).join(', ') + '] rad<br>⟨Z₀⟩ = ' + r.expval_z0.toFixed(3);
  $('modelTable').innerHTML = '<tr><th>Model</th><th>Type</th><th>Says</th><th>Chance of busy</th><th>ms</th></tr>' + r.predictions.map((p, i) => `
    <tr class="${i === 0 ? 'hl' : ''}"><td>${p.model}${i === 0 ? ' (ours)' : ''}</td><td><span class="tag ${p.family[0]}">${p.family}</span></td>
    <td>${busyPill(p.p > 0.5)}</td><td><div class="barrow"><div class="bar"><i style="width:${100 * p.p}%;background:${p.family === 'quantum' ? css('--q') : css('--c')}"></i></div>${pct(p.p, 0)}</div></td>
    <td class="num">${p.ms.toFixed(2)}</td></tr>`).join('');
  $('timing').textContent = `All five models together answered in ${r.total_ms.toFixed(1)} ms (the limit is 1000 ms).`;

  const s = r.similar;
  if (!s.n) {
    $('similarSummary').innerHTML = '<p class="nodata">No real 2017–18 hour had exactly these conditions (for example, snow at 30 °C). Try other settings.</p>';
    $('similarTable').innerHTML = '';
  } else {
    const share = s.busy / s.n;
    const agree = (share >= 0.5) === busy;
    $('similarSummary').innerHTML = `<div class="truthrow">
      <div><b>${num(s.n)}</b>real hours with these conditions</div>
      <div><b>${num(s.busy)}</b>of them were really busy (${pct(share, 0)})</div>
      <div><b>${num(s.avg_cars)}</b>cars on average</div>
      <div><b class="q">${s.vqc_correct} / ${s.n}</b>quantum model right on those hours</div></div>
      <p class="verdict ${agree ? 'ok' : 'bad'}">${agree ? '✓' : '✗'} The prediction “${r.decision}” ${agree ? 'agrees' : 'does not agree'} with what usually happened
        in these conditions (${share >= 0.5 ? 'mostly busy' : 'mostly not busy'}).</p>`;
    $('similarTable').innerHTML = '<tr><th>Real date</th><th>Time</th><th>Weather</th><th>Temp.</th><th>Cars really counted</th><th>Correct answer</th><th class="qh">Quantum model said</th><th class="qh"></th></tr>' +
      s.examples.map((e) => `<tr class="${(e.p.vqc > 0.5) === (e.actual === 1) ? '' : 'wrong'}"><td>${e.weekday} ${e.date}</td><td>${e.time}</td><td>${e.weather}</td><td>${e.temp_c} °C</td>
        <td class="num">${num(e.cars)}</td><td>${busyPill(e.actual)}</td><td>${busyPill(e.p.vqc > 0.5)}</td><td>${mark((e.p.vqc > 0.5) === (e.actual === 1))}</td></tr>`).join('');
  }

  const labels = r.state_probs.map((_, i) => '|' + i.toString(2).padStart(4, '0') + '⟩');
  chart('state', $('stateChart'), { type: 'bar', data: { labels, datasets: [{ data: r.state_probs,
    backgroundColor: labels.map((_, i) => ((i >> 3) & 1) ? css('--bad') : css('--q')) }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'probability' } } } } });
}

// ---------- 4. detour planner ----------
const rP = seg('rP');
let rSeed = 0;
for (let i = 0; i < 8; i++) $('rEmerg').innerHTML += `<option value="${i}">Stream ${i + 1}</option>`;
$('rHour').oninput = () => $('rHourOut').textContent = hh($('rHour').value);
$('rNew').onclick = () => { rSeed++; runRoute(); };
$('rRun').onclick = runRoute;
const road = (x) => `<span class="pill ${x ? 'alt' : 'main'}">${x ? 'Detour' : 'Highway'}</span>`;

async function runRoute() {
  $('rRun').textContent = 'Planning…';
  const r = await api('/api/route', { hour: +$('rHour').value, emergency: +$('rEmerg').value, p: rP(), seed: rSeed });
  $('rRun').textContent = 'Plan detours';
  const q = r.plans.QAOA, e = r.plans.Exact, max = Math.max(...r.demands);
  const same = Math.abs(q.cost - e.cost) < 1e-9;
  $('plan').innerHTML = `<p class="verdict ${same ? 'ok' : 'bad'}">${same
      ? `✓ The quantum plan is the best possible plan (rank 1 of ${r.n_plans}).`
      : `The quantum plan ranks #${r.rank} of ${r.n_plans}; its delay is ${pct((q.cost - e.cost) / e.cost, 1)} above the best plan.`}</p>
    <table class="models"><tr><th>Stream</th><th>Cars</th><th></th><th class="qh">Quantum plan</th><th>Best plan (checked)</th><th></th></tr>` +
    r.demands.map((d, i) => `<tr><td>Stream ${i + 1}${i === r.emergency ? ' 🚑' : ''}</td><td class="num">${num(d)}</td>
      <td><div class="bar"><i style="width:${100 * d / max}%;background:${css('--muted')}"></i></div></td>
      <td>${road(q.x[i])}</td><td>${road(e.x[i])}</td><td>${mark(q.x[i] === e.x[i])}</td></tr>`).join('') + '</table>' +
    `<table class="models" style="margin-top:14px"><tr><th>Method</th><th>Delay score J (lower = better)</th><th>Cars sent on detour</th><th>Time</th></tr>` +
    [['QAOA', 'Quantum (QAOA)'], ['Exact', 'Check all 256 plans (classical)'], ['Greedy', 'Quick rule of thumb (classical)']].map(([k, n]) => {
      const v = r.plans[k];
      return `<tr class="${k === 'QAOA' ? 'hl' : ''}"><td>${n}</td><td class="num">${v.cost.toFixed(4)}</td><td class="num">${pct(v.alt_share, 0)}</td><td class="num">${v.ms.toFixed(1)} ms</td></tr>`;
    }).join('') + '</table>' +
    `<p class="muted">Safety check: the ambulance stream ${q.x[r.emergency] ? '✗ was sent on the detour, so the plan is rejected' : '✓ stays on the cleared highway'}.</p>`;
  chart('qaoa', $('qaoaChart'), { type: 'bar', data: { labels: r.top_states.map((s) => [...s.bits].map((b) => (b === '1' ? 'D' : 'H')).join('')),
    datasets: [{ data: r.top_states.map((s) => s.prob), backgroundColor: r.top_states.map((s) =>
      Math.abs(s.cost - e.cost) < 1e-9 ? css('--ok') : css('--q')) }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'chance the quantum algorithm outputs this plan' }, ticks: { callback: (v) => pct(v, 1) } },
        y: { ticks: { font: { family: 'JetBrains Mono' } } } } } });
  $('qaoaInfo').innerHTML = `Each row is a plan for streams 1–8 (H = highway, D = detour). <b style="color:${css('--ok')}">Green</b> = the best possible plan
    (shown even if it is not among the most likely). A random guess would pick it 1 time in 256 (0.39%); this quantum circuit outputs it
    ${pct(r.p_opt, 1)} of the time, ${(r.p_opt / (1 / 256)).toFixed(1)}× more often. Like a real quantum computer, we run the circuit 256 times
    and keep the plan with the lowest delay. Deeper circuits (larger p) concentrate more chance on good plans.`;
}

// ---------- 5. error correction ----------
let tQ;
$('pErr').oninput = () => { $('pOut').textContent = ($('pErr').value / 10).toFixed(1) + '%'; clearTimeout(tQ); tQ = setTimeout(runQEC, 200); };
async function runQEC() {
  const p = $('pErr').value / 1000;
  const r = await api('/api/qec?p=' + p);
  const names = { unprotected: 'No protection (1 qubit)', bit_flip: 'Bit-flip code (3)', phase_flip: 'Phase-flip code (3)', shor: 'Shor code (9)' };
  const protects = { unprotected: 'nothing', bit_flip: 'X flips only', phase_flip: 'Z flips only', shor: 'X, Y and Z' };
  chart('qec', $('qecChart'), { type: 'bar', data: { labels: Object.values(names),
    datasets: [{ label: 'read back wrong', data: Object.keys(names).map((k) => r[k].p_logical),
      backgroundColor: Object.keys(names).map((k) => r[k].p_logical < r.unprotected.p_logical ? css('--ok') : k === 'unprotected' ? css('--muted') : css('--bad')) }] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => pct(c.raw, 2) + ' of runs read back wrong' } } },
      scales: { y: { title: { display: true, text: 'share of runs read back wrong' }, ticks: { callback: (v) => pct(v, 1) } } } } });
  $('qecTable').innerHTML = '<tr><th>Code</th><th>Qubits used</th><th>Fixes</th><th>Read back wrong</th></tr>' + Object.keys(names).map((k) =>
    `<tr class="${k === 'shor' ? 'hl' : ''}"><td>${names[k]}</td><td class="num">${r[k].qubits}</td><td>${protects[k]}</td><td class="num">${pct(r[k].p_logical, 2)}</td></tr>`).join('');
  const u = r.unprotected.p_logical, s = r.shor.p_logical;
  $('qecInfo').innerHTML = `Green = better than no protection, red = worse. ` + (s < u
    ? `At ${pct(p, 1)} noise the Shor code cuts errors from ${pct(u, 2)} to ${pct(s, 2)}.`
    : `At ${pct(p, 1)} noise there are so many mistakes that even the Shor code cannot keep up: it only helps below roughly 8%.`);
}

// ---------- 6. all results ----------
function renderResults() {
  const models = ['Re-uploading VQC (ours)', 'Standard VQC', 'Quantum-kernel SVM', 'Logistic regression', 'SVM (RBF)', 'Random forest', 'MLP (16-16)'];
  const colors = ['#7c3aed', '#c4b5fd', '#a78bfa', '#94a3b8', '#38bdf8', '#0284c7', '#475569'];
  const sizes = [...new Set(R.comparison.map((r) => r.n))];
  chart('lc', $('lcChart'), { type: 'line', data: { labels: sizes, datasets: models.map((m, i) => ({ label: m,
    data: sizes.map((n) => { const a = R.comparison.filter((r) => r.model === m && r.n === n).map((r) => r.acc); return a.reduce((s, v) => s + v, 0) / a.length; }),
    borderColor: colors[i], backgroundColor: colors[i], borderWidth: i === 0 ? 3 : 1.5, borderDash: i > 0 && i < 3 ? [4, 3] : [] })) },
    options: { plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: { x: { title: { display: true, text: 'training hours' } }, y: { title: { display: true, text: 'test accuracy' } } } } });

  const groups = {};
  R.ablation.forEach((r) => (groups[r.variant] ??= []).push(r));
  $('ablTable').innerHTML = '<tr><th>Variant</th><th>settings</th><th>accuracy</th><th>AUC</th></tr>' + Object.entries(groups).map(([v, rs]) => {
    const m = (k) => rs.reduce((s, r) => s + r[k], 0) / rs.length;
    return `<tr class="${v.includes('ours') ? 'hl' : ''}"><td>${v.trim()}</td><td class="num">${rs[0].n_params}</td><td class="num">${pct(m('acc'))}</td><td class="num">${m('auc').toFixed(3)}</td></tr>`;
  }).join('');

  const qg = {};
  R.qaoa.forEach((r) => (qg[r.method] ??= []).push(r));
  const f = (rs, k, fn) => rs[0][k] === undefined ? '—' : fn(rs.reduce((s, r) => s + r[k], 0) / rs.length);
  $('qaoaTable').innerHTML = '<tr><th>Method</th><th>⟨r⟩</th><th>gap of best plan</th><th>ms</th></tr>' + Object.entries(qg).map(([m, rs]) =>
    `<tr class="${m.startsWith('QAOA') ? 'hl' : ''}"><td>${m}</td><td class="num">${f(rs, 'approx_ratio', (v) => v.toFixed(3))}</td>
     <td class="num">${f(rs, 'gap', (v) => pct(v, 2))}</td><td class="num">${f(rs, 'ms', (v) => v.toFixed(1))}</td></tr>`).join('');
}

loadOverview().then(predict).then(() => openTab(location.hash.slice(1) || 'start'));
