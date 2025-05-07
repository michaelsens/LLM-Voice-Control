const { chromium } = require("playwright");
const commands = require("./commands.json");

(async () => {
  const browser = await chromium.launch({
    headless: false,
    args: [
      `--disable-extensions-except=./extension`,
      `--load-extension=./extension`
    ]
  });
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto("https://example.com/login");

  async function sendRpc(rpc) {
    const id = Math.floor(Math.random() * 1e9);
    return page.evaluate(({ rpc, id }) => {
      return new Promise(resolve => {
        window.addEventListener("message", function handler(e) {
          if (e.data.rpcTarget === "voice2rpc-response" && e.data.id === id) {
            window.removeEventListener("message", handler);
            resolve(e.data.result);
          }
        });
        window.postMessage({ rpcTarget: "voice2rpc", rpc, id }, "*");
      });
    }, { rpc, id });
  }

  for (const rpc of commands) {
    console.log("Sending RPC:", rpc);
    const resp = await sendRpc(rpc);
    console.log("→ Response:", resp);
  }

  await browser.close();
})();
