chrome.action.onClicked.addListener(tab => {
  chrome.tabs.sendMessage(tab.id, { action: "start-listening" });
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  console.log("background got RPC:", msg, "from tab", sender.tab?.id);
  (async () => {
    try {
      const tabId = sender.tab.id;
      switch (msg.method) {
        case "list_elements": {
          const [res] = await chrome.scripting.executeScript({
            target: {tabId},
            func: () =>
              Array.from(document.querySelectorAll("*")).map(el => ({
                selector:
                  el.tagName.toLowerCase() +
                  (el.id ? `#${el.id}` : "") +
                  (el.className ? `.${[...el.classList].join(".")}` : ""),
                role: el.getAttribute("role") || el.tagName.toLowerCase(),
                name: (el.innerText || el.alt || "").trim().slice(0, 30)
              }))
          });
          sendResponse({ elements: res.result });
          break;
        }
        case "click":
          await chrome.scripting.executeScript({
            target: {tabId},
            args: [msg.params.selector],
            func: sel => document.querySelector(sel)?.click()
          });
          sendResponse({ status: "ok" });
          break;
        case "type":
          await chrome.scripting.executeScript({
            target: {tabId},
            args: [msg.params.selector, msg.params.text],
            func: (sel,txt) => { const e=document.querySelector(sel); if(e)e.value=txt; }
          });
          sendResponse({ status: "ok" });
          break;
        default:
          console.warn("unknown method:", msg.method);
          sendResponse({ error: "unknown method" });
      }
    } catch (err) {
      console.error("background error:", err);
      sendResponse({ error: err.message });
    }
  })();
  return true;
});
