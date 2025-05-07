const { chromium } = require("playwright");
const repl         = require("repl");

(async () => {
  const browser = await chromium.connectOverCDP("http://localhost:9222");
  console.log("Connected to Chrome over CDP on port 9222");

  function getPage() {
    for (const ctx of browser.contexts()) {
      for (const pg of ctx.pages()) {
        const url = pg.url();
        if (url.startsWith("http://") || url.startsWith("https://")) {
          return pg;
        }
      }
    }
    throw new Error("No HTTP(S) pages found—navigate to an http(s) URL and retry.");
  }

  async function sendRpc(rpc) {
    const page = getPage();
    console.log(`[sendRpc] → ${rpc.method} on ${page.url()}`);
    const id = Date.now();
    return page.evaluate(({ rpc, id }) => {
      return new Promise(resolve => {
        window.addEventListener("message", function handler(e) {
          if (
            e.data.rpcTarget === "voice2rpc-response" &&
            e.data.id === id
          ) {
            window.removeEventListener("message", handler);
            resolve(e.data.result);
          }
        });
        window.postMessage({ rpcTarget: "voice2rpc", rpc, id }, "*");
      });
    }, { rpc, id });
  }

  const server = repl.start({ prompt: "rpc> " });
  server.context.sendRpc = sendRpc;
})();
