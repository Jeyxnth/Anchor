// Ad-hoc demo walkthrough driver — follows DEMO_SCRIPT.md's actual order
// live against the running dashboard. Not part of the app.
import puppeteer from "puppeteer-core";

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT_DIR = "D:\\Anchor\\frontend\\_demo";

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  defaultViewport: { width: 1680, height: 1200 },
});

try {
  const page = await browser.newPage();
  await page.goto("http://localhost:5173/", { waitUntil: "networkidle0", timeout: 20000 });

  // Step 0: set the scene — fresh baseline experiment, 200 rows (covers all
  // three demo transaction IDs: TXN000000, TXN000025, TXN002166 < 200).
  await page.waitForSelector("#rows-input", { timeout: 10000 });
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
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: `${OUT_DIR}\\0_dashboard_top.png`, clip: { x: 0, y: 0, width: 1680, height: 1400 } });

  async function openCase(txnId, outName) {
    // Close any open modal first
    const closeBtn = await page.$(".modal-header button");
    if (closeBtn) {
      await closeBtn.click();
      await new Promise((r) => setTimeout(r, 200));
    }
    await page.evaluate((id) => {
      const btn = [...document.querySelectorAll(".demo-card")].find((b) => b.textContent.includes(id));
      if (btn) btn.click();
    }, txnId);
    await page.waitForSelector(".modal", { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 400));
    const modalHeight = await page.evaluate(() => document.querySelector(".modal").scrollHeight);
    const el = await page.$(".modal");
    await el.screenshot({ path: `${OUT_DIR}\\${outName}.png` });
    console.log(`${txnId} -> ${outName}.png (modal height ${modalHeight}px)`);
  }

  // Step 1: TXN000000 — normal case, all 5 actions eligible.
  await openCase("TXN000000", "1_normal_case");

  // Step 2: TXN000025 — opted-out, compliance restricts to no_action only.
  await openCase("TXN000025", "2_opted_out");

  // Step 3: TXN002166 — contact-capped, restricted to escalate/no_action.
  await openCase("TXN002166", "3_contact_capped");

  console.log("done");
} finally {
  await browser.close();
}
