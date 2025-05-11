/* background.js */

// Send; return true if a listener existed, false if not.
async function safeSend(tabId, payload) {
  try {
    await chrome.tabs.sendMessage(tabId, payload);
    return true;
  } catch (err) {
    if (err.message.includes("Receiving end does not exist")) {
      console.warn("No content-script in tab", tabId, "– message dropped");
      return false;               // not fatal
    }
    throw err;                    // some other kind of failure
  }
}

/* Start speech recognition when the extension icon is clicked */
chrome.action.onClicked.addListener(async tab => {
  await safeSend(tab.id, { action: "start-listening" });
});


// Main message handler
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // ──────────────────────────────────────────────────────────
  //  0. Speech transcript ➜ local LLM (port 6006) ➜ RPC
  // ──────────────────────────────────────────────────────────
  if (msg.action === "transcript") {
    (async () => {
      try {
        // Send the transcript to the locally running LLM
        const llmResp = await fetch("http://127.0.0.1:6006/infer", {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ text: msg.text })
        });

        if (!llmResp.ok) {
          // Print full error body for easier debugging
          const bodyText = await llmResp.text();
          console.error("LLM server error", llmResp.status, bodyText);
          throw new Error(`LLM server HTTP ${llmResp.status}`);
        }

        const rpc = await llmResp.json();      // model should return { method, params }
        console.log("LLM → RPC", rpc);

        if (!rpc?.method) throw new Error("LLM reply missing 'method' field");

        // Forward the RPC to the active tab
        const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!activeTab) throw new Error("No active tab to receive RPC");

        const delivered = await safeSend(activeTab.id, { rpc });
        if (!delivered) {                 // no content.js in that tab
          sendResponse({ error: "no-receiver" });
          return;                         // stop processing this transcript
        }

        sendResponse({ status: "ok" });
      } catch (err) {
        console.error("transcript→LLM pipeline error:", err);
        sendResponse({ error: err.message });
      }
    })();

    // Keep the sendResponse channel open for the async work above
    return true;
  }

  // ──────────────────────────────────────────────────────────
  //  1-11. Existing RPC calls (unchanged)
  // ──────────────────────────────────────────────────────────
  console.log("background got RPC:", msg);

  (async () => {
    try {
      const [activeTab] = await chrome.tabs.query({
        active: true,
        currentWindow: true
      });
      if (!activeTab) throw new Error("No active tab");
      const currentTabId = activeTab.id;

      switch (msg.method) {
        case "navigate": {
          let url = String(msg.params.url || "").trim();
          if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
          await chrome.tabs.update(currentTabId, { url });
          sendResponse({ status: "ok" });
          break;
        }

        case "openTab": {
          const newTab = await chrome.tabs.create({
            url: "https://www.google.com",
            active: true
          });
          sendResponse({ status: "ok", tabId: newTab.id });
          break;
        }

        case "closeTab": {
          const targetId = Number.isFinite(msg.params.tabId)
            ? msg.params.tabId
            : currentTabId;
          try {
            await chrome.tabs.remove(targetId);
            sendResponse({ status: "ok" });
          } catch {
            sendResponse({ error: `cannot close tab ${targetId}` });
          }
          break;
        }

        case "switchTab": {
          const index = Number.isFinite(msg.params.index)
            ? msg.params.index
            : null;
          if (index === null) {
            sendResponse({ error: "missing tab index" });
            break;
          }
          const tabs = await chrome.tabs.query({ currentWindow: true });
          const target = tabs[index];
          if (target) {
            await chrome.tabs.update(target.id, { active: true });
            sendResponse({ status: "ok" });
          } else {
            sendResponse({ error: `no tab at index ${index}` });
          }
          break;
        }

        case "click": {
          const text = String(msg.params.text || "");
          await chrome.scripting.executeScript({
            target: { tabId: currentTabId },
            args: [text],
            func: searchText => {
              const sel = [
                "button",
                "a",
                "[role=button]",
                "input[type=button]",
                "input[type=submit]"
              ].join(",");
              const els = Array.from(document.querySelectorAll(sel));
              const match = els.find(el => {
                const candidates = [
                  el.innerText,
                  el.value,
                  el.getAttribute("aria-label"),
                  el.getAttribute("title")
                ];
                return candidates
                  .filter(Boolean)
                  .some(str =>
                    str.toLowerCase().includes(searchText.toLowerCase())
                  );
              });
              if (match) {
                match.scrollIntoView({ block: "center" });
                match.click();
              }
            }
          });
          sendResponse({ status: "ok" });
          break;
        }

        case "type": {
          const text = String(msg.params.text || "");
          const field = String(msg.params.field || "");
          await chrome.scripting.executeScript({
            target: { tabId: currentTabId },
            args: [text, field],
            func: (value, fieldDesc) => {
              const selector = [
                "input:not([type=hidden])",
                "textarea",
                "[contenteditable='true']",
                "[role='textbox']"
              ].join(",");
              const candidates = Array.from(
                document.querySelectorAll(selector)
              );

              function getLabels(el) {
                const labels = [];
                if (el.placeholder) labels.push(el.placeholder);
                if (el.title) labels.push(el.title);
                if (el.getAttribute("aria-label"))
                  labels.push(el.getAttribute("aria-label"));
                if (el.name) labels.push(el.name);
                if (el.id) labels.push(el.id);
                const lab =
                  document.querySelector(`label[for="${el.id}"]`) ||
                  el.closest("label");
                if (lab?.innerText) labels.push(lab.innerText);
                return labels.map(s => s.toLowerCase());
              }

              let match = candidates.find(el => {
                const labels = getLabels(el);
                return labels.some(lbl =>
                  lbl.includes(fieldDesc.toLowerCase())
                );
              });

              // fallback for Google main search box
              if (!match && fieldDesc.toLowerCase().includes("search")) {
                match = document.querySelector('input[name="q"]');
              }

              if (match) {
                match.scrollIntoView({ block: "center" });
                match.focus();
                if (match.isContentEditable) match.innerText = value;
                else match.value = value;
                match.dispatchEvent(new Event("input", { bubbles: true }));
                match.dispatchEvent(new Event("change", { bubbles: true }));
              }
            }
          });
          sendResponse({ status: "ok" });
          break;
        }

        case "scroll": {
          const direction = String(msg.params.direction || "down");
          const amount = Number.isFinite(msg.params.amount)
            ? msg.params.amount
            : 300;
          await chrome.scripting.executeScript({
            target: { tabId: currentTabId },
            args: [direction, amount],
            func: (dir, amt) => {
              const deltas = {
                up: [0, -amt],
                down: [0, amt],
                left: [-amt, 0],
                right: [amt, 0]
              };
              const [dx, dy] = deltas[dir] || [0, 0];
              window.scrollBy(dx, dy);
            }
          });
          sendResponse({ status: "ok" });
          break;
        }

        case "reload": {
          await chrome.tabs.reload(currentTabId);
          sendResponse({ status: "ok" });
          break;
        }

        case "search": {
          const q = String(msg.params.query || "");
          const url = `https://www.google.com/search?q=${encodeURIComponent(q)}`;
          await chrome.tabs.create({ url });
          sendResponse({ status: "ok" });
          break;
        }

        case "goBack": {
          const steps = Number.isFinite(msg.params.steps)
            ? msg.params.steps
            : 1;
          await chrome.scripting.executeScript({
            target: { tabId: currentTabId },
            args: [steps],
            func: count => {
              for (let i = 0; i < count; i++) history.back();
            }
          });
          sendResponse({ status: "ok" });
          break;
        }

        case "goForward": {
          const steps = Number.isFinite(msg.params.steps)
            ? msg.params.steps
            : 1;
          await chrome.scripting.executeScript({
            target: { tabId: currentTabId },
            args: [steps],
            func: count => {
              for (let i = 0; i < count; i++) history.forward();
            }
          });
          sendResponse({ status: "ok" });
          break;
        }

        default:
          sendResponse({ error: "unknown method" });
      }
    } catch (err) {
      console.error("background error:", err);
      sendResponse({ error: err.message });
    }
  })();

  return true; // keep the message channel open for async response
});
