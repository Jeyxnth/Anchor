// One-off diagnostic script — takes a screenshot of the dashboard with real
// data loaded (types a batch id into the Policy Comparison loader and
// clicks Load) via puppeteer-core driving the machine's existing Chrome.
// Not part of the app; deleted after use.
import puppeteer from "puppeteer-core";

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT_PATH = process.argv[2] || "D:\\Anchor\\frontend\\_shot.png";
const BATCH_ID = process.argv[3] || "baseline_experiment_full";

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  defaultViewport: { width: 1680, height: 1200 },
});

try {
  const page = await browser.newPage();
  await page.goto("http://localhost:5173/", { waitUntil: "networkidle0", timeout: 20000 });

  // Small row count -> a short Recent Decisions table, so a targeted
  // screenshot is actually legible. Run a fresh baseline experiment (not
  // "Load") so metrics/decisions/comparison ALL populate.
  await page.waitForSelector('#rows-input', { timeout: 10000 });
  await page.evaluate(() => {
    const el = document.getElementById("rows-input");
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(el, "200");
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "Run Baseline Experiment");
    if (btn) btn.click();
  });

  await page.waitForFunction(
    () => document.body.innerText.includes("same population across all three policies"),
    { timeout: 60000 }
  );
  await page.waitForFunction(
    () => document.body.innerText.includes("shown") && document.body.innerText.includes("Recent Decisions"),
    { timeout: 20000 }
  );
  await new Promise((r) => setTimeout(r, 500));

  const base = OUT_PATH.replace(/\.png$/, "");
  const pageHeight = await page.evaluate(() => document.body.scrollHeight);
  await page.screenshot({ path: `${base}_top.png`, clip: { x: 0, y: 0, width: 1680, height: 1400 } });
  await page.screenshot({ path: `${base}_mid.png`, clip: { x: 0, y: 1400, width: 1680, height: Math.min(1300, pageHeight - 1400) } });

  // Also open the compliance-changed trace modal (TXN000025, opted-out) for review.
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll(".demo-card")].find((b) => b.textContent.includes("TXN000025"));
    if (btn) btn.click();
  });
  await page.waitForSelector(".modal", { timeout: 10000 });
  await new Promise((r) => setTimeout(r, 400));
  await page.screenshot({ path: `${base}_modal.png`, clip: { x: 400, y: 0, width: 880, height: 1400 } });

  console.log("saved shots, pageHeight:", pageHeight);
} finally {
  await browser.close();
}
