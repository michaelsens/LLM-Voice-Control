console.log("content.js running on", location.href);

window.addEventListener("message", e => {
  if (e.source !== window || e.data?.rpcTarget !== "voice2rpc") return;
  console.log("content.js forwarding RPC:", e.data.rpc);
  chrome.runtime.sendMessage(e.data.rpc, resp => {
    console.log("content.js got response:", resp);
    window.postMessage({
      rpcTarget: "voice2rpc-response",
      id:        e.data.id,
      result:    resp
    },"*");
  });
});

chrome.runtime.onMessage.addListener((msg,_,sendResponse) => {
  if (msg.action === "start-listening") startRecognition();
});

let statusEl;
function makeStatusEl() {
  const el = document.createElement("div");
  Object.assign(el.style, {
    position:    "fixed",
    bottom:      "1em",
    right:       "1em",
    background:  "rgba(0,0,0,0.7)",
    color:       "white",
    padding:     "0.5em 1em",
    borderRadius:"4px",
    zIndex:      999999
  });
  document.body.appendChild(el);
  return el;
}
function setStatus(txt) {
  if (!statusEl) statusEl = makeStatusEl();
  statusEl.textContent = txt;
}

function startRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    console.error("No Web Speech API here");
    return;
  }
  const r = new SR();
  r.continuous     = false;
  r.interimResults = false;
  r.lang           = "en-US";

  r.onstart   = () => { console.log("🔊 Recognition started"); setStatus("Listening…"); };
  r.onresult  = evt => {
    const t = evt.results[0][0].transcript.trim();
    console.log("You said:", t);
    setStatus(`Heard: “${t}”`);

  };
  r.onerror   = e => { console.error("Recognition error:", e.error); setStatus(`Error: ${e.error}`); };
  r.onend     = () => { console.log("Recognition ended"); setStatus("Idle"); };

  try { r.start(); }
  catch (err) { console.warn("recog.start() failed", err); }
}
