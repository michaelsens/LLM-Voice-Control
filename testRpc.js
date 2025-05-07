const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.connectOverCDP("http://localhost:9222");

  const context = browser.contexts()[0];
  const page = context.pages().find(p => {
    const url = p.url();
    return url.startsWith("http://") || url.startsWith("https://");
  });
  if (!page) {
    console.error("No HTTP(S) pages found—please open one and retry.");
    process.exit(1);
  }
  console.log("Sending list_elements to", page.url());

  const result = await page.evaluate(() => {
    return new Promise(resolve => {
      const id = Date.now();
      window.addEventListener("message", function handler(e) {
        if (e.data.rpcTarget === "voice2rpc-response" && e.data.id === id) {
          window.removeEventListener("message", handler);
          resolve(e.data.result);
        }
      });
      window.postMessage(
        { rpcTarget: "voice2rpc", rpc: { method: "list_elements", params: {} }, id },
        "*"
      );
    });
  });

  console.log("RPC result:", result);

  await browser.close();
})();
