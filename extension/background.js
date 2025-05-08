// background.js

// When the toolbar icon is clicked, tell the page to start listening
chrome.action.onClicked.addListener(tab => {
  chrome.tabs.sendMessage(tab.id, { action: "start-listening" });
});

// Handle all incoming RPC calls
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  console.log("background got RPC:", msg, "from tab", sender.tab?.id);

  (async () => {
    try {
      const currentTabId = sender.tab.id;

      switch (msg.method) {

        // 1. navigate
        case "navigate": {
          const url = String(msg.params.url || "");
          await chrome.tabs.update(currentTabId, { url });
          sendResponse({ status: "ok" });
          break;
        }

        // 2. openTab — always open a new tab to Google.com
        case "openTab": {
          const newTab = await chrome.tabs.create({
            url: "https://www.google.com",
            active: true
          });
          sendResponse({ status: "ok", tabId: newTab.id });
          break;
        }

        // 3. closeTab
        case "closeTab": {
          const targetId = Number.isFinite(msg.params.tabId)
                         ? msg.params.tabId
                         : currentTabId;
          try {
            await chrome.tabs.remove(targetId);
            sendResponse({ status: "ok" });
          } catch (err) {
            sendResponse({ error: `cannot close tab ${targetId}` });
          }
          break;
        }

        // 4. switchTab (by index only)
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

        // 5. click — match any clickable element by visible text
        case "click": {
          const text = String(msg.params.text || "");
          await chrome.scripting.executeScript({
            target: { tabId: currentTabId },
            args: [text],
            func: (searchText) => {
              const sel = [
                "button",
                "a",
                "[role=button]",
                "input[type=button]",
                "input[type=submit]"
              ].join(",");
              const elements = Array.from(document.querySelectorAll(sel));
              const match = elements.find(el => {
                const candidates = [
                  el.innerText,
                  el.value,
                  el.getAttribute("aria-label"),
                  el.getAttribute("title"),
                ];
                return candidates
                  .filter(Boolean)
                  .some(str => str.toLowerCase().includes(searchText.toLowerCase()));
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

        // 6. type — find input/textarea/contenteditable by label/placeholder/aria-label/name/id
        case "type": {
          const text  = String(msg.params.text  || "");
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
              const candidates = Array.from(document.querySelectorAll(selector));

              function getLabels(el) {
                const labels = [];
                if (el.placeholder)      labels.push(el.placeholder);
                if (el.title)            labels.push(el.title);
                if (el.getAttribute("aria-label")) labels.push(el.getAttribute("aria-label"));
                if (el.name)             labels.push(el.name);
                if (el.id)               labels.push(el.id);
                const lab = document.querySelector(`label[for="${el.id}"]`) 
                          || el.closest("label");
                if (lab?.innerText)      labels.push(lab.innerText);
                return labels.map(s => s.toLowerCase());
              }

              let match = candidates.find(el => {
                const labels = getLabels(el);
                return labels.some(lbl => lbl.includes(fieldDesc.toLowerCase()));
              });

              if (!match && fieldDesc.toLowerCase().includes("search")) {
                match = document.querySelector('input[name="q"]');
              }

              if (match) {
                match.scrollIntoView({ block: "center" });
                match.focus();

                if (match.isContentEditable) {
                  match.innerText = value;
                } else {
                  match.value = value;
                }

                // dispatch events so React/Vue/etc. pick it up
                match.dispatchEvent(new Event("input",  { bubbles: true }));
                match.dispatchEvent(new Event("change", { bubbles: true }));
              }
            }
          });
          sendResponse({ status: "ok" });
          break;
        }


        // 7. scroll
        case "scroll": {
          const direction = String(msg.params.direction || "down");
          const amount    = Number.isFinite(msg.params.amount)
                          ? msg.params.amount
                          : 300;
          await chrome.scripting.executeScript({
            target: { tabId: currentTabId },
            args: [direction, amount],
            func: (dir, amt) => {
              const deltas = {
                up:    [0, -amt],
                down:  [0,  amt],
                left:  [-amt,0],
                right: [amt,  0]
              };
              const [dx, dy] = deltas[dir] || [0, 0];
              window.scrollBy(dx, dy);
            }
          });
          sendResponse({ status: "ok" });
          break;
        }

        // 8. reload — default browser reload
        case "reload": {
          await chrome.tabs.reload(currentTabId);
          sendResponse({ status: "ok" });
          break;
        }


        // 9. search — always Google, new tab
        case "search": {
          const query = String(msg.params.query || "");
          const url = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
          await chrome.tabs.create({ url });
          sendResponse({ status: "ok" });
          break;
        }

        // 10. goBack
        case "goBack": {
          const steps = Number.isFinite(msg.params.steps) ? msg.params.steps : 1;
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

        // 11. goForward
        case "goForward": {
          const steps = Number.isFinite(msg.params.steps) ? msg.params.steps : 1;
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

  return true;
});
