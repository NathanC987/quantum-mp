const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (v, d = 1) => (100 * v).toFixed(d) + '%';
const num = (v) => Math.round(v).toLocaleString('en-US');
const hh = (h) => String(h).padStart(2, '0') + ':00';
const charts = {};
let R = null, O = null, THR = 3340;

const ramps = (bits) => [...bits].map((b) => b === '1' ? '<i class="on">●</i>' : '<i>○</i>').join('');
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
  if (name === 'noise' && !charts.noiseAcc) renderNoise();
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
  const ee = Object.fromEntries(R.endtoend.map((r) => [r.policy, r]));
  $('kpis').innerHTML = [
    ['q', pct(s.vqc.acc), `The quantum model was right in <b>${num(s.vqc.tp + s.vqc.tn)} of ${num(total)}</b> test hours it had never seen`],
    ['c', pct(bestC.acc), `Best ordinary (classical) model, ${bestC.model}, on the same hours. The two are on par`],
    ['q', R.latency['VQC – exact simulator'].median_ms.toFixed(1) + ' ms', 'Time for one quantum prediction (the limit is 1 second = 1000 ms)'],
    ['q', pct(ee['Re-uploading VQC (ours)'].share_of_possible), `Of the largest possible delay reduction, achieved by detour plans built from the quantum prediction (fixed plan: ${pct(ee['Always plan for busy'].share_of_possible, 0)})`],
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
  const sv = d.savings;
  $('daySummary').innerHTML = `<div><b class="q">${nQ} / ${rows.length}</b> hours right: quantum model</div>
    <div><b class="c">${nC} / ${rows.length}</b> hours right: classical MLP</div>
    <div><b>${rows.filter((r) => r.actual).length}</b> hours were really busy</div>
    <div><b class="q">${pct(sv.saved, 0)}</b> less delay with our detours (best possible: ${pct(sv.best_saved, 0)})</div>`;

  chart('day', $('dayChart'), { data: { labels: rows.map((r) => r.time), datasets: [
    { type: 'bar', label: 'Cars really counted', data: rows.map((r) => r.cars), borderRadius: 3,
      backgroundColor: rows.map((r) => ok(r, 'vqc') ? css('--ok') : css('--bad')) },
    { type: 'line', label: 'Busy line', data: rows.map(() => THR), borderColor: css('--text'), borderDash: [6, 4], borderWidth: 1.2, pointRadius: 0 },
  ] }, options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: {
      label: (c) => c.datasetIndex ? `Busy line: ${num(THR)}` : `Real: ${num(c.raw)} cars (${rows[c.dataIndex].actual ? 'busy' : 'not busy'})`,
      afterLabel: (c) => c.datasetIndex ? '' : `Quantum model: ${pct(rows[c.dataIndex].p.vqc, 0)} chance of busy ${ok(rows[c.dataIndex], 'vqc') ? '✓' : '✗'}` } } },
    scales: { y: { title: { display: true, text: 'cars that hour' } } } } });

  $('dayTable').innerHTML = `<tr><th>Time</th><th>Weather</th><th>Temp.</th><th>Cars really counted</th><th>Correct answer</th>
    <th class="qh">Quantum model says</th><th class="qh"></th><th>Classical MLP says</th><th></th><th class="qh">Detour ramps 1–8</th><th>Delay: none → ours (best)</th></tr>` + rows.map((r) => {
    const pq = r.p.vqc > 0.5, pc = r.p.mlp > 0.5;
    return `<tr class="${ok(r, 'vqc') ? '' : 'wrong'}"><td>${r.time}</td><td>${r.weather}</td><td>${r.temp_c} °C</td>
      <td class="num">${num(r.cars)}</td><td>${busyPill(r.actual)}</td>
      <td>${busyPill(pq)} <span class="muted">${pct(r.p.vqc, 0)}</span></td><td>${mark(ok(r, 'vqc'))}</td>
      <td>${busyPill(pc)} <span class="muted">${pct(r.p.mlp, 0)}</span></td><td>${mark(ok(r, 'mlp'))}</td>
      <td class="ramps">${ramps(r.ramps)}</td><td class="num">${r.delay.never.toFixed(2)} → <b>${r.delay.ours.toFixed(2)}</b> <span class="muted">(${r.delay.best.toFixed(2)})</span></td></tr>`;
  }).join('');
}

// ---------- 3. predict & plan: one input for both ----------
const dayType = seg('dayType', predict);
const rP = seg('rP', predict);
$('emerg').innerHTML = '<option value="-1">none</option>' + [...Array(8).keys()].map((i) => `<option value="${i}">Ramp ${i + 1}</option>`).join('');
let tPred, busyReq = 0;
['hour', 'temp'].forEach((id) => $(id).oninput = () => { clearTimeout(tPred); tPred = setTimeout(predict, 150); });
['weather', 'emerg', 'hw'].forEach((id) => $(id).onchange = predict);
const road = (x) => `<span class="pill ${x ? 'alt' : 'main'}">${x ? 'Detour' : 'Highway'}</span>`;

async function predict() {
  $('hourOut').textContent = hh($('hour').value);
  $('tempOut').textContent = $('temp').value + ' °C';
  const req = ++busyReq;
  $('plan').classList.add('loading');
  const r = await api('/api/plan', { hour: +$('hour').value, day_type: dayType(), temp_c: +$('temp').value, weather: +$('weather').value,
    emergency: +$('emerg').value, p: rP(), noise_level: +$('hw').value });
  if (req !== busyReq) return;                       // a newer request is on its way
  $('plan').classList.remove('loading');
  const rt = r.route, busy = r.decision === 'Busy';
  const d = $('decision'); d.innerHTML = `${r.decision} <small>${pct(rt.p_busy, 0)} chance of more than ${num(r.threshold)} cars</small>`;
  d.className = 'decision ' + (busy ? 'yes' : 'no');
  $('hwNote').innerHTML = r.hardware ? `<p class="hwnote">On the noisy chip the quantum model read <b>${pct(r.hardware.p_raw, 0)}</b>;
    after readout correction and rescaling: <b>${pct(r.hardware.p_mitigated, 0)}</b> (noise-free: ${pct(r.predictions[0].p, 0)}).
    The plan uses the corrected value.</p>` : '';
  $('modelTable').innerHTML = '<tr><th>Model (noise-free)</th><th>Type</th><th>Says</th><th>Chance of busy</th><th>ms</th></tr>' + r.predictions.map((p, i) => `
    <tr class="${i === 0 ? 'hl' : ''}"><td>${p.model}${i === 0 ? ' (ours)' : ''}</td><td><span class="tag ${p.family[0]}">${p.family}</span></td>
    <td>${busyPill(p.p > 0.5)}</td><td><div class="barrow"><div class="bar"><i style="width:${100 * p.p}%;background:${p.family === 'quantum' ? css('--q') : css('--c')}"></i></div>${pct(p.p, 0)}</div></td>
    <td class="num">${p.ms.toFixed(2)}</td></tr>`).join('');
  $('timing').textContent = `Prediction and detour plan together took ${r.total_ms.toFixed(0)} ms (the limit is 1000 ms).`;
  $('angles').innerHTML = 'clues as angles x = [' + r.angles.map((a) => a.toFixed(3)).join(', ') + '] rad<br>⟨Z₀⟩ = ' + r.expval_z0.toFixed(3);

  // step 2: detour plan
  const cars = rt.ramp_cars, max = Math.max(...cars);
  if (!rt.released) {
    $('plan').innerHTML = `<p class="verdict ok">No detours: the hour is predicted <b>not busy</b>, so every ramp stays on the highway.
      The quantum search is not run (the classical decision layer only releases detours for busy hours).</p>` + rampTable(cars, max, null, null, rt.shares);
    if (charts.qaoa) { charts.qaoa.destroy(); delete charts.qaoa; }
    $('qaoaInfo').textContent = 'No detour search for hours predicted not busy.';
  } else {
    const q = rt.plans.QAOA, e = rt.plans.Exact, same = Math.abs(q.cost - e.cost) < 1e-9;
    const saved = (rt.no_detour_cost - q.cost) / rt.no_detour_cost;
    $('plan').innerHTML = `<p class="pplan">Expected traffic ≈ <b>${num(rt.expected_cars)} cars</b> this hour
        (weighted by the ${pct(rt.p_busy, 0)} chance of busy). Planned to cut expected delay by <b>${pct(saved, 0)}</b> versus no detours.</p>
      <p class="verdict ${same ? 'ok' : 'bad'}">${same ? `✓ The quantum plan is the best possible plan (rank 1 of ${rt.n_plans}).`
        : `The quantum plan ranks #${rt.rank} of ${rt.n_plans}; its expected delay is ${pct((q.cost - e.cost) / e.cost, 1)} above the best plan.`}</p>` +
      rampTable(cars, max, q.x, e.x, rt.shares, rt.emergency) +
      `<table class="models" style="margin-top:14px"><tr><th>Method</th><th>Expected delay (lower = better)</th><th>Time</th></tr>` +
      [['QAOA', `Quantum (QAOA, ${+$('hw').value ? 'noisy chip' : 'ideal'})`], ['Exact', 'Check all 256 plans (classical)'], ['Greedy', 'Quick rule of thumb (classical)']].map(([k, n]) =>
        `<tr class="${k === 'QAOA' ? 'hl' : ''}"><td>${n}</td><td class="num">${rt.plans[k].cost.toFixed(4)}</td><td class="num">${rt.plans[k].ms.toFixed(1)} ms</td></tr>`).join('') +
      `<tr><td>No detours</td><td class="num">${rt.no_detour_cost.toFixed(4)}</td><td></td></tr></table>` +
      `<p class="muted">${rt.emergency === null ? 'No ambulance.' : `Safety check: the ambulance at ramp ${rt.emergency + 1} ${q.x[rt.emergency] ? '✗ was sent on the detour, so the plan is rejected' : '✓ stays on the cleared highway'}.`}</p>`;
    chart('qaoa', $('qaoaChart'), { type: 'bar', data: { labels: rt.top.map((t) => [...t.bits].map((b) => (b === '1' ? 'D' : 'H')).join('')),
      datasets: [{ data: rt.top.map((t) => t.prob), backgroundColor: rt.top.map((t) => t.best ? css('--ok') : css('--q')) }] },
      options: { maintainAspectRatio: false, indexAxis: 'y', plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: 'chance the circuit outputs this plan' }, ticks: { callback: (v) => pct(v, 1) } },
          y: { ticks: { font: { family: 'JetBrains Mono' } } } } } });
    $('qaoaInfo').innerHTML = `Each row is a plan for ramps 1–8 (H = highway, D = detour). <b style="color:${css('--ok')}">Green</b> = the best possible plan.
      A random guess would pick it 1 time in 256 (0.39%); this circuit outputs it ${pct(rt.p_opt, 1)} of the time. Like a real quantum computer,
      we run it 256 times and keep the plan with the lowest expected delay.${+$('hw').value ? ' On the noisy chip the outputs are spread more evenly, so good plans come up less often.' : ''}`;
  }

  // ground truth
  const s = r.similar;
  if (!s.n) {
    $('similarSummary').innerHTML = '<p class="nodata">No real 2017–18 hour had exactly these conditions (for example, snow at 30 °C). Try other settings.</p>';
    $('similarTable').innerHTML = '';
  } else {
    const share = s.busy / s.n, agree = (share >= 0.5) === busy;
    $('similarSummary').innerHTML = `<div class="truthrow">
      <div><b>${num(s.n)}</b>real hours with these conditions</div>
      <div><b>${num(s.busy)}</b>of them were really busy (${pct(share, 0)})</div>
      <div><b class="q">${s.vqc_correct} / ${s.n}</b>quantum model right on those hours</div>
      <div><b class="q">${pct(s.saved, 0)}</b>less delay in those hours with this plan (best possible: ${pct(s.best_saved, 0)})</div></div>
      <p class="verdict ${agree ? 'ok' : 'bad'}">${agree ? '✓' : '✗'} The prediction “${r.decision}” ${agree ? 'agrees' : 'does not agree'} with what usually happened
        in these conditions (${share >= 0.5 ? 'mostly busy' : 'mostly not busy'}, ${num(s.avg_cars)} cars on average).</p>`;
    $('similarTable').innerHTML = '<tr><th>Real date</th><th>Time</th><th>Weather</th><th>Temp.</th><th>Cars really counted</th><th>Correct answer</th><th class="qh">Quantum model said</th><th class="qh"></th></tr>' +
      s.examples.map((e) => `<tr class="${(e.p.vqc > 0.5) === (e.actual === 1) ? '' : 'wrong'}"><td>${e.weekday} ${e.date}</td><td>${e.time}</td><td>${e.weather}</td><td>${e.temp_c} °C</td>
        <td class="num">${num(e.cars)}</td><td>${busyPill(e.actual)}</td><td>${busyPill(e.p.vqc > 0.5)}</td><td>${mark((e.p.vqc > 0.5) === (e.actual === 1))}</td></tr>`).join('');
  }

  const labels = r.state_probs.map((_, i) => '|' + i.toString(2).padStart(4, '0') + '⟩');
  chart('state', $('stateChart'), { type: 'bar', data: { labels, datasets: [{ data: r.state_probs,
    backgroundColor: labels.map((_, i) => ((i >> 3) & 1) ? css('--bad') : css('--q')) }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'probability' } } } } });
}

function rampTable(cars, max, xq, xe, shares, emergency = null) {
  return `<table class="models"><tr><th>On-ramp</th><th>Share</th><th>Expected cars</th><th></th>${xq ? '<th class="qh">Quantum plan</th><th>Best plan (checked)</th><th></th>' : '<th>Plan</th>'}</tr>` +
    cars.map((c, i) => `<tr><td>Ramp ${i + 1}${i === emergency ? ' 🚑' : ''}</td><td class="num">${pct(shares[i], 0)}</td><td class="num">${num(c)}</td>
      <td><div class="bar"><i style="width:${100 * c / max}%;background:${css('--muted')}"></i></div></td>
      ${xq ? `<td>${road(xq[i])}</td><td>${road(xe[i])}</td><td>${mark(xq[i] === xe[i])}</td>` : `<td>${road(0)}</td>`}</tr>`).join('') + '</table>';
}

// ---------- 4. hardware noise ----------
function renderNoise() {
  const nz = R.noise, lv = nz.map((r) => r.level + '×');
  const line = (label, data, color, dash) => ({ label, data, borderColor: color, backgroundColor: color, borderDash: dash || [], tension: 0.2 });
  const opts = (y, fmt) => ({ maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 12 } } },
    scales: { x: { title: { display: true, text: 'noise level (1× = today’s chips, 0 = ideal)' } }, y: { title: { display: true, text: y }, ticks: { callback: fmt } } } });
  chart('noiseAcc', $('noiseAccChart'), { type: 'line', data: { labels: lv, datasets: [
    line('accuracy, raw', nz.map((r) => r.acc_raw), css('--muted')),
    line('accuracy, mitigated', nz.map((r) => r.acc_mitigated), css('--q')),
    line('share of possible delay saving (mitigated)', nz.map((r) => r.saving_mitigated), css('--ok'), [5, 4]),
  ] }, options: opts('', (v) => pct(v, 1)) });
  const n1 = nz.find((r) => r.level === 1), n4 = nz.find((r) => r.level === 4);
  $('noiseAccInfo').innerHTML = `500 test hours, 1,024 shots each. At today’s noise level the accuracy is
    ${pct(n1.acc_mitigated)} (noise-free: ${pct(R.noise_ideal_acc)}); even at 4× it is ${pct(n4.acc_mitigated)}. Busy/not-busy decisions survive noise well
    because noise mostly shrinks the reading towards 0 without moving the decision boundary.`;
  chart('noiseProb', $('noiseProbChart'), { type: 'line', data: { labels: lv, datasets: [
    line('raw', nz.map((r) => r.prob_err_raw), css('--muted')),
    line('readout corrected', nz.map((r) => r.prob_err_readout), css('--c')),
    line('readout corrected + rescaled', nz.map((r) => r.prob_err_mitigated), css('--q')),
  ] }, options: opts('error in chance of busy', (v) => pct(v, 0)) });
  const g = {};
  R.qaoa_noise.forEach((r) => (g[r.level] ??= []).push(r));
  const lvls = Object.keys(g).map(Number).sort((a, b) => a - b);
  const avg = (rs, k) => rs.reduce((s, r) => s + r[k], 0) / rs.length;
  const rnd = avg(R.qaoa.filter((r) => r.method === 'Random guess (p=0)'), 'approx_ratio');
  chart('noiseQaoa', $('noiseQaoaChart'), { type: 'line', data: { labels: lvls.map((l) => l + '×'), datasets: [
    line('QAOA p=2 quality', lvls.map((l) => avg(g[l], 'approx_ratio')), css('--q')),
    line('random plans', lvls.map(() => rnd), css('--muted'), [5, 4]),
  ] }, options: opts('approximation ratio (1 = always best plan)', (v) => v.toFixed(2)) });
  const loop = R.qaoa_in_loop, same = (k) => loop.filter((l) => l[k]).length / loop.length;
  $('noiseQaoaInfo').innerHTML = `Quality of the plans the circuit outputs (1 = always the best plan). Noise pushes QAOA towards random guessing.
    Because each output plan is checked classically and the best of 256 is kept, the final plan is still good: on ${loop.length} busy test hours the
    chosen plan was the exact best ${pct(same('same_ideal'), 0)} of the time without noise and ${pct(same('same_noisy'), 0)} on today’s noisy chip;
    the other plans were near-ties, and the delay they really caused differed from the exact plan’s by less than
    ${pct(Math.max(0.001, Math.abs(loop.reduce((s, l) => s + l.extra_noisy, 0)) / loop.reduce((s, l) => s + l.exact, 0)), 1)}.
    At 8 ramps even random sampling would find such near-ties, so QAOA has no advantage here.`;
  if (R.qaoa_circuit) $('qaoaCx').textContent = R.qaoa_circuit.cx;
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
