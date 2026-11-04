// Browser end-to-end check of the read-only routes against a deployed (or local static) site.
// Usage: NODE_PATH=$(npm root -g) node scripts/e2e/routes.js https://host/base/ [shots_dir]   (needs playwright)
// Exit code 0 only if every route renders the stored real outputs with no console errors or failed requests.
const { chromium } = require('playwright');
(async () => {
  const [base, shots] = process.argv.slice(2);
  if (!base) { console.error('usage: node routes.js <base-url-with-trailing-slash> [shots_dir]'); process.exit(2); }
  const root = base.replace(/\/$/, '');
  const run = 'demo_developmental_100a_80y_s42';
  const routes = [
    ['/', /Human Possibility Engine/],
    [`/lab/society?run=${run}`, /Stored real output/],
    [`/lab/runs/${run}/agents/a00010`, /Acquired skills/],
    ['/lab/learning', /Exp 1|H1/],
    ['/lab/experiment', /Exp 2/],
    ['/lab/evidence', /claim/i],
    ['/lab/benchmarks', /agent-years|Benchmarks/],
    ['/future/welcome', /ILLUSTRATIVE SCENARIO|Illustrative/i],
    ['/demo', /INVESTOR WALKTHROUGH/i],
  ];
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1360, height: 900 }, deviceScaleFactor: 1 });
  const problems = [];
  page.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text().slice(0, 200)}`); });
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message.slice(0, 200)}`));
  page.on('response', (r) => { if (r.status() >= 400 && r.url().startsWith(root)) problems.push(`http ${r.status()} ${r.url()}`); });
  let failed = 0;
  for (const [r, expect] of routes) {
    const before = problems.length;
    try {
      await page.goto(root + r, { waitUntil: 'networkidle', timeout: 90000 });
      await page.waitForTimeout(2500);
    } catch (e) { problems.push(`goto ${r}: ${e.message.slice(0, 120)}`); }
    const text = (await page.innerText('body').catch(() => '')).replace(/\s+/g, ' ');
    const ok = expect.test(text) && problems.length === before;
    if (!ok) failed++;
    if (shots) await page.screenshot({ path: `${shots}/${(r.replace(/[^a-z0-9]+/gi, '_') || 'root').slice(0, 40)}.png` });
    console.log(`${ok ? 'OK  ' : 'FAIL'} ${r} | new problems ${problems.length - before} | ${text.slice(0, 110)}`);
  }
  console.log('PROBLEMS', JSON.stringify(problems.slice(0, 20), null, 1));
  console.log(`SUMMARY ${routes.length - failed}/${routes.length} routes ok`);
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
