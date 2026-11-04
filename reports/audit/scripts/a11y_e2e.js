// Audit 7: Playwright + axe-core over the main routes against a live API + web preview.
// Usage: node a11y_e2e.js <web_base> <axe.min.js path> <out.json> [shots_dir]
const path = require('path');
const globalRoot = require('child_process').execSync('npm root -g').toString().trim();
const { chromium } = require(path.join(globalRoot, 'playwright'));
const fs = require('fs');

(async () => {
  const [base, axePath, outPath, shots] = process.argv.slice(2);
  const axeSrc = fs.readFileSync(axePath, 'utf8');
  const run = 'demo_developmental_100a_80y_s42';
  const routes = [
    '/', `/lab/society?run=${run}`, `/lab/runs/${run}/agents/a00010`, '/lab/learning', '/lab/experiment',
    '/lab/evidence', '/lab/benchmarks', '/future/welcome', '/demo',
  ];
  const browser = await chromium.launch();
  const out = { base, routes: {} };
  for (const scheme of ['light', 'dark']) {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 }, colorScheme: scheme });
    const page = await ctx.newPage();
    for (const r of routes) {
      const problems = [];
      page.removeAllListeners('console'); page.removeAllListeners('response'); page.removeAllListeners('pageerror');
      page.on('console', (m) => { if (m.type() === 'error') problems.push('console: ' + m.text().slice(0, 160)); });
      page.on('pageerror', (e) => problems.push('pageerror: ' + e.message.slice(0, 160)));
      page.on('response', (resp) => { if (resp.status() >= 400 && resp.url().includes(':87')) problems.push(`http ${resp.status()} ${resp.url()}`); });
      await page.goto(base + r, { waitUntil: 'networkidle', timeout: 60000 });
      await page.waitForTimeout(1200);
      await page.addScriptTag({ content: axeSrc });
      const res = await page.evaluate(async () => {
        // eslint-disable-next-line no-undef
        const r = await axe.run(document, { resultTypes: ['violations'] });
        return r.violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.length, sample: v.nodes.slice(0, 2).map((n) => n.target.join(' ')) }));
      });
      const text = (await page.innerText('body')).replace(/\s+/g, ' ');
      const mockBanner = text.includes('MOCK DATA');
      const key = `${scheme} ${r}`;
      out.routes[key] = { violations: res, problems, mockBanner, chars: text.length };
      if (shots && scheme === 'light') await page.screenshot({ path: `${shots}/${r.replace(/[^a-z0-9]+/gi, '_').slice(0, 40) || 'root'}.png` });
      console.log(key, '| axe:', res.map((v) => `${v.id}(${v.impact},${v.nodes})`).join(' ') || 'none', '| problems:', problems.length, '| mock:', mockBanner);
    }
    // mobile overflow check
    await ctx.close();
  }
  const mctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const mp = await mctx.newPage();
  out.mobile_overflow = {};
  for (const r of routes) {
    await mp.goto(base + r, { waitUntil: 'networkidle', timeout: 60000 });
    await mp.waitForTimeout(800);
    out.mobile_overflow[r] = await mp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  }
  console.log('mobile horizontal overflow px:', JSON.stringify(out.mobile_overflow));
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  await browser.close();
})();
