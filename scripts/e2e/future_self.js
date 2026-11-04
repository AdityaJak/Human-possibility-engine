// Future Self end-to-end (consent -> decision -> scenarios) against a static deployment; runs the real
// Pyodide engine in the browser. Usage: NODE_PATH=$(npm root -g) node scripts/e2e/future_self.js https://host/base/ [shots_dir]
// Exit code 0 only if at least two "Illustrative scenario" labels render and no probability language appears.
const { chromium } = require('playwright');
(async () => {
  const [base, shots] = process.argv.slice(2);
  if (!base) { console.error('usage: node future_self.js <base-url-with-trailing-slash> [shots_dir]'); process.exit(2); }
  const root = base.replace(/\/$/, '');
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1360, height: 900 } });
  const problems = []; const t0 = Date.now();
  p.on('console', (m) => { if (m.type() === 'error') problems.push('console: ' + m.text().slice(0, 160)); });
  p.on('pageerror', (e) => problems.push('pageerror: ' + e.message.slice(0, 160)));
  p.on('response', (r) => { if (r.status() >= 400 && r.url().startsWith(root)) problems.push(`http ${r.status()} ${r.url()}`); });
  const log = (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
  await p.goto(root + '/future/welcome', { waitUntil: 'networkidle', timeout: 90000 });
  { const lk = p.getByRole('link', { name: /continue to consent/i }); if (await lk.count()) await lk.first().click(); else await p.getByRole('button', { name: /continue to consent/i }).click(); }
  log('consent page: ' + p.url());
  // the consent step renders after a client-side navigation; wait for it instead of counting too early
  await p.locator('input[type=checkbox]').first().waitFor({ timeout: 60000 });
  const boxes = p.locator('input[type=checkbox]'); const n = await boxes.count();
  for (let i = 0; i < n; i++) await boxes.nth(i).check();
  await p.locator('button.btn--future').first().click(); log('consent submitted');
  await p.waitForTimeout(800);
  for (let k = 0; k < 3; k++) {
    if (await p.locator('#q').count()) break;
    const btn = p.getByRole('button', { name: /continue|next|decision/i }).first();
    const link = p.getByRole('link', { name: /continue|next|decision/i }).first();
    if (await btn.count()) await btn.click(); else if (await link.count()) await link.click(); else await p.goto(root + '/future/decision');
    await p.waitForTimeout(800);
  }
  log('decision page: ' + p.url());
  await p.getByRole('button', { name: /use the job \/ startup example/i }).click();
  await p.getByRole('button', { name: /explore possible futures/i }).click(); log('submitted; waiting for engine');
  let status = '';
  const deadline = Date.now() + 240000;
  while (Date.now() < deadline) {
    const txt = (await p.innerText('body')).replace(/\s+/g, ' ');
    const s = txt.match(/Loading the scenario engine[^.]*\.?|Building scenarios…|Installing[^.]*\.|Downloading[^.]*\./);
    if (s && s[0] !== status) { status = s[0]; log('status: ' + status.slice(0, 120)); }
    if (/Illustrative scenario/i.test(txt) && (await p.locator('article, .card').count()) > 0 && !/Building scenarios…/.test(txt)) break;
    if (/scenario engine error|Cannot reach|not available/i.test(txt)) { log('ERROR TEXT: ' + txt.slice(0, 300)); break; }
    await p.waitForTimeout(1000);
  }
  const body = (await p.innerText('body')).replace(/\s+/g, ' ');
  const cards = (body.match(/Illustrative scenario/gi) || []).length;
  const prob = body.match(/\d+(\.\d+)?\s*%|\bpercent\b|\bodds\b|\bprobability of\b|\bguarantee|you will become/i);
  log(`scenario labels: ${cards}; forbidden language: ${prob ? prob[0] : 'none'}; url ${p.url()}`);
  if (shots) await p.screenshot({ path: shots + '/future_scenarios.png', fullPage: false });
  const perf = await p.evaluate(() => performance.getEntriesByType('resource').filter((r) => /pyodide|\.whl|\.wasm|\.zip/.test(r.name)).map((r) => ({ n: r.name.split('/').pop(), kb: Math.round((r.transferSize || r.encodedBodySize || 0) / 1024) })));
  log('engine resources: ' + JSON.stringify(perf.slice(0, 12)));
  console.log('PROBLEMS', JSON.stringify(problems.slice(0, 10)));
  await b.close();
  process.exit(cards >= 2 && !prob ? 0 : 1);
})();
