const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const STARTERS = [
  { text: "My sink is broken. What should I do?" },
  { text: "Can the rent go up in the middle of a lease?" },
  { text: "The tenant is two weeks late on rent. What happens next?" },
  { text: "A lease says: “No pets of any kind are permitted.” Is that enforceable?", clause: "No pets of any kind are permitted." },
  { text: "We're negotiating who pays to fix the broken fridge. What should I bring up?", needsRole: true },
];

let status = {};

// ---------- status ----------
async function loadStatus() {
  try {
    status = await (await fetch("/api/status")).json();
  } catch {
    $("#status").innerHTML = `<span class="pill warn">Server offline</span>`;
    return status;
  }
  const pills = [];
  $("#beta").hidden = !status.hosted;
  if (status.hosted) pills.push(`<span class="pill warn" title="A test version: answers are still being reviewed">Beta</span>`);
  else if (status.preview) pills.push(`<span class="pill warn" title="Answers are placeholders built from the sources; the model isn't loaded">Preview mode</span>`);
  else pills.push(`<span class="pill ${status.fine_tuned ? "on" : "warn"}" title="${status.fine_tuned ? "Our model, trained on Ontario sources" : "Training not finished: using the untrained base model"}">${status.fine_tuned ? "Ontario model" : "Base model (untrained)"}</span>`);
  pills.push(`<span class="pill ${status.jev ? "on" : ""}" title="${status.jev ? "Jev scans lease clauses first" : "Add TYPESAFE_API_KEY to .env to scan leases with Jev"}">Jev ${status.jev ? "on" : "off"}</span>`);
  $("#status").innerHTML = pills.join("");
  $("#privacy").textContent = status.hosted
    ? "What you type or upload is read on our server and isn't saved. With names, emails, phone numbers and addresses removed, it's sent to TypeSafe AI (Jev) to pick and check the answer."
    : status.jev
      ? "Answers are written on this computer. With Jev on, clause text is sent to TypeSafe AI to be scanned."
      : "Everything runs on this computer. The lease isn't sent anywhere.";
  return status;
}

// ---------- tabs ----------
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll(".tab")) {
      const on = t === tab;
      t.setAttribute("aria-selected", on);
      $("#" + t.getAttribute("aria-controls")).hidden = !on;
    }
  });
}

// ---------- streaming ----------
async function streamAnswer(payload, onSources, onToken, signal, onCheck = () => {}, onRoute = () => {}, onSteps = () => {}) {
  const res = await fetch("/api/ask", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal,
  });
  if (!res.ok) {
    const err = new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    err.status = res.status;
    throw err;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const event = /^event: (.*)$/m.exec(chunk)?.[1];
      const data = JSON.parse(/^data: (.*)$/m.exec(chunk)?.[1] ?? "null");
      if (event === "sources") onSources(data);
      else if (event === "token") onToken(data);
      else if (event === "check") onCheck(data);
      else if (event === "route") onRoute(data);
      else if (event === "steps") onSteps(data);
      else if (event === "error") throw new Error(data);
      else if (event === "done") return;
    }
  }
}

// ---------- answer rendering ----------
// The model quotes in a fixed shape: `says: "…"`, `Why: "…"`. A quote opens after a colon and
// closes at a quote mark that ends the line, so quote marks inside the passage are safe.
const QUOTE_RE = /(:[ \n])"([\s\S]+?)"(?=[ \t]*(?:\n|$))/g;
const normQuotes = (t) => t.replaceAll("“", '"').replaceAll("”", '"');

// Pointers: {2.3} = source 2, sentence 3; {2.3-5} = sentences 3 to 5. The model writes these
// instead of retyping quotes, and the exact text is filled in here from the source itself.
const POINTER_RE = /\{(\d)\.(\d+)(?:-(?:\d\.)?(\d+))?\}/g;

function resolvePointer(sources, n, a, b) {
  const src = sources[n - 1];
  if (!src?.sentences) return null;
  const from = +a, to = +(b ?? a);
  if (from < 1 || to < from || to > src.sentences.length) return null;
  return src.sentences.slice(from - 1, to).join(" ");
}

// A heading taken from the quoted source's own title, so it can't disagree with the quote.
function sourceTitle(src) {
  const l = src.label;
  let m;
  if (src.kind === "rta" && (m = /s\. ([\d.]+): (.*)/.exec(l))) return `RTA s. ${m[1]} · ${m[2]}`;
  if (src.kind === "lease" && (m = /Part ([A-U]): (.*?)( \(Part.*)?$/.exec(l))) return `Standard lease, Part ${m[1]} · ${m[2]}`;
  if (src.kind === "lease" && (m = /Section (\d+): (.*)$/.exec(l))) return `Standard lease, Section ${m[1]} · ${m[2]}`;
  if (src.kind === "mine") return l;
  if (src.kind === "guide") {
    const [doc, heading = ""] = l.split(', "');
    const g = /Interpretation Guideline (\d+)/.exec(doc);
    return `${g ? "LTB Guideline " + g[1] : doc.replace("LTB brochure: ", "LTB brochure, ")} · ${heading.replace(/"$/, "").split(":")[0]}`;
  }
  return l;
}

const POINTER_LINE = /^(- )?\*\*.+?\*\*[^\n{]*?(\{(\d)\.\d+(?:-(?:\d\.)?\d+)?\})\s*$/;

// Model-written headings on quote lines are replaced with the source's title, and repeated
// lines (a small model can loop) are dropped.
const CASE_LINE = /^(- )?In \[(\d)\][^\n]*?(?:(Why|What was ordered):\s*(\{[^}]+\}))?\s*$/;
const OUTCOME_TEXT = { tenant: "decided the case in the tenant's favour", landlord: "decided the case in the landlord's favour" };

// Make every line that refers to a source say exactly what that source is. Decision lines are
// rebuilt from the decision itself (a model can copy a citation from an earlier answer); lines
// with a pointer to nothing, and repeated lines, are dropped.
function restructure(text, sources) {
  const seen = new Set();
  const complete = text.replace(/\{[\d.\-]*$/, "");
  return complete.split("\n").filter((line) => {
    const k = line.trim();
    if (k && seen.has(k)) return false;
    seen.add(k);
    return ![...line.matchAll(POINTER_RE)].some((m) => resolvePointer(sources, m[1], m[2], m[3]) === null);
  }).map((line) => {
    const c = CASE_LINE.exec(line.trim());
    const src = c && sources[c[2] - 1];
    if (src?.kind === "case") {
      const ptr = c[4]?.startsWith(`{${c[2]}.`) ? c[4] : null;
      return `${c[1] ?? ""}In [${c[2]}] ${src.label}, the Board ${OUTCOME_TEXT[src.side] ?? "made the order below"}.${ptr ? ` ${c[3]}: ${ptr}` : ""}`;
    }
    const m = POINTER_LINE.exec(line.trim());
    if (!m || !sources[m[3] - 1]) return line;
    return `${m[1] ?? ""}**${sourceTitle(sources[m[3] - 1])}.** ${m[2]}`;
  }).join("\n");
}

function quotesIn(text, sources = []) {
  const fromPointers = [...text.matchAll(POINTER_RE)].map((m) => resolvePointer(sources, m[1], m[2], m[3])).filter(Boolean);
  return fromPointers.concat([...normQuotes(text).matchAll(QUOTE_RE)].map((m) => m[2].replace(/\s*…$/, "")));
}

function shortLabel(src) {
  if (src.kind === "mine") return "Your lease";
  if (src.kind === "rta") return "RTA s. " + (/s\. ([\d.]+)/.exec(src.label)?.[1] ?? "");
  if (src.kind === "lease") return /Section (\d+)/.test(src.label) ? "Standard lease, s. " + /Section (\d+)/.exec(src.label)[1]
                                                                 : "Standard lease, Part " + (/Part ([A-U])/.exec(src.label)?.[1] ?? "");
  if (src.kind === "guide") {
    const g = /Interpretation Guideline (\d+)/.exec(src.label);
    return g ? `LTB Guideline ${g[1]}` : "LTB brochure";
  }
  return "LTB " + (/decision (.+?) \(/.exec(src.label)?.[1] ?? "decision");
}

const KIND_NAME = { rta: "Statute", lease: "Standard lease", guide: "LTB guidance", case: "LTB decision", mine: "Your lease" };

function inline(s, kinds) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\s)_(.+?)_(?=\s|$)/g, '$1<em class="muted">$2</em>')
    .replace(/\[(\d)\]/g, (m, n) => kinds[n] ? `<button class="cite ${kinds[n]}" data-n="${n}" title="Show source ${n}">${n}</button>` : m);
}

function renderAnswer(text, sources, reviewed = false) {
  const kinds = Object.fromEntries(sources.map((s) => [s.n, s.kind]));
  // Reviewed playbook answers keep their headings; model-written headings become source titles.
  let t = (reviewed ? normQuotes(text) : restructure(normQuotes(text), sources)).replace(/\{[\d.\-]*$/, "");
  if (!sources.length) t = normQuotes(text);
  // Pull quotes out first (they can span lines), render the rest line by line, then put them back.
  const quotes = [];
  const pointed = [];
  t = t.replace(POINTER_RE, (m, n, a, b) => {
    pointed.push({ n: +n, ptr: m, text: resolvePointer(sources, n, a, b) });
    return `\u0002${pointed.length - 1}\u0002`;
  });
  t = t.replace(QUOTE_RE, (m, lead, body) => { quotes.push(body); return `${lead.trimEnd()}\u0000${quotes.length - 1}\u0000`; });
  const open = /(:[ \n])"([\s\S]*)$/.exec(t);  // a quote still streaming in
  let openQuote = null;
  if (open) { openQuote = open[2]; t = t.slice(0, open.index) + ":\u0001"; }

  const html = [];
  let list = false;
  for (const raw of t.split("\n")) {
    const line = raw.trim();
    const bullet = /^- /.test(line);
    if (bullet && !list) { html.push("<ul>"); list = true; }
    if (!bullet && list) { html.push("</ul>"); list = false; }
    if (!line) continue;
    const body = inline(esc(bullet ? line.slice(2) : line), kinds);
    const cls = /^\*\*Short answer\.\*\*/.test(line) ? ' class="short-answer"' : /^\*\*In your lease\.\*\*/.test(line) ? ' class="in-lease"' : "";
    html.push(bullet ? `<li>${body}</li>` : `<p${cls}>${body}</p>`);
  }
  if (list) html.push("</ul>");
  return html.join("")
    .replace(/\u0002(\d+)\u0002/g, (m, i) => {
      const p = pointed[i];
      if (!p.text) return `<q class="unverified">This points at a passage that isn't in the sources.</q>`;
      return `<span class="pq" data-n="${p.n}" data-ptr="${p.ptr}"><q>${esc(p.text)}</q><span class="q-src">${esc(shortLabel(sources[p.n - 1]))}</span></span>`;
    })
    .replace(/\u0000(\d+)\u0000/g, (m, i) => `<q>${esc(quotes[i])}</q>`)
    .replace("\u0001", openQuote !== null ? `<q class="open">${esc(openQuote)}</q>` : "");
}

function sourceChips(sources) {
  return `<div class="sources-label">Sources the answer is based on</div>` + sources.map((src) => {
    const side = src.kind === "case" && ["tenant", "landlord"].includes(src.side) ? `<span class="side">${src.side} won</span>` : "";
    return `<button class="source-chip ${src.kind}" data-n="${src.n}" title="${esc(src.label)}"><span class="n">${src.n}</span><span class="name">${esc(shortLabel(src))}</span>${side}</button>`;
  }).join("");
}

async function runAnswer(container, payload, signal) {
  const block = document.createElement("div");
  block.className = "answer";
  block.innerHTML = `<div class="answer-text streaming"><p class="thinking">Reading the law…</p></div><div class="next-steps" hidden></div><div class="check-note" hidden></div><div class="sources"></div>`;
  container.replaceChildren(block);
  const out = $(".answer-text", block);
  const srcBox = $(".sources", block);
  let sources = [], answer = "", reviewed = false, scenario = null;

  const paint = () => {
    out.innerHTML = renderAnswer(answer, sources, reviewed);
    const cited = new Set([...answer.matchAll(/\[(\d)\]/g)].map((m) => m[1]));
    for (const chip of srcBox.querySelectorAll(".source-chip")) chip.classList.toggle("cited", cited.has(chip.dataset.n));
  };
  block.addEventListener("click", (e) => {
    const n = e.target.closest("[data-n]")?.dataset.n;
    if (n) openSource(sources[n - 1], quotesIn(answer, sources));
  });

  let check = null;
  try {
    await streamAnswer(payload, (s) => { sources = s; srcBox.innerHTML = sourceChips(s); },
      (tok) => { answer += tok; paint(); }, signal, (c) => { check = c; },
      (r) => {
        if ("selected" in r) {  // passages picked by Jev (0 means none fitted; the answer says so)
          if (r.selected > 0) block.insertAdjacentHTML("afterbegin", `<div class="route-note">Passages picked by Jev from official sources</div>`);
          return;
        }
        if (!r.topic) return;
        reviewed = true;
        scenario = r.scenario ?? null;
        block.insertAdjacentHTML("afterbegin", `<div class="route-note">From a reviewed guide: ${esc(r.topic.split(" (")[0])}</div>`);
      }, (steps) => showSteps(block, steps));
  } catch (err) {
    out.classList.remove("streaming");
    if (err.name === "AbortError") return { state: "aborted", answer };
    console.warn("answer failed:", err);  // details for debugging; people see a plain message
    out.innerHTML = `<p class="muted">${[429, 413].includes(err.status) ? esc(err.message)
      : "Couldn't answer that just now. Please try again in a moment."}</p>`;
    return { state: "error", answer };
  }
  out.classList.remove("streaming");
  paint();
  if (check) showCheck(block, check);
  if (payload.mode === "question") feedbackRow(block, payload.text, scenario);
  return { state: "done", answer, scenario };
}

// Practical next steps under a guide's answer, and a written request to start from.
function showSteps(block, data) {
  const box = $(".next-steps", block);
  const name = { tenant: "If you're the tenant", landlord: "If you're the landlord" };
  const sides = Object.entries(data.sides);
  box.innerHTML = `<h4>What you can do next</h4>` + sides.map(([side, steps]) =>
    (sides.length > 1 ? `<div class="side-label">${name[side]}</div>` : "") +
    `<ol>${steps.map((t) => `<li>${esc(t)}</li>`).join("")}</ol>`).join("");
  const letter = data.letter && typeof LETTERS !== "undefined" && LETTERS[data.letter];
  if (letter) {
    const btn = document.createElement("button");
    btn.className = "btn ghost small";
    btn.type = "button";
    btn.textContent = `Draft a written ${letter.title.toLowerCase()}`;
    btn.onclick = () => {
      const date = new Date().toLocaleDateString("en-CA", { year: "numeric", month: "long", day: "numeric" });
      const wrap = document.createElement("div");
      wrap.className = "letter";
      wrap.innerHTML = `<p class="muted">A starting point to edit: fill in the brackets, then send it to the landlord and keep a copy. Nothing is sent from here.</p>
        <textarea rows="16" aria-label="${esc(letter.title)}"></textarea><button class="btn ghost small" type="button">Copy</button>`;
      // "my main door is broken what to do?" → "My main door is broken."
      let issue = data.issue.trim().replace(/[\s,.;:-]*(what (should|can|do) (i|we) do( now| about it)?|what to do|what now|what are my options|help( me)?|any advice)\s*[?.!]*$/i, "")
        .replace(/[?.!\s]+$/, "");
      issue = issue ? issue[0].toUpperCase() + issue.slice(1) + "." : "[describe the problem]";
      $("textarea", wrap).value = letter.body(issue, date);
      $("button", wrap).onclick = async (e) => {
        try { await navigator.clipboard.writeText($("textarea", wrap).value); e.target.textContent = "Copied"; }
        catch { $("textarea", wrap).select(); }
      };
      btn.replaceWith(wrap);
    };
    box.append(btn);
  }
  box.hidden = false;
}

// Thumbs up/down under each chat answer. On the hosted version this is how testing turns into fixes.
function feedbackRow(block, question, scenario) {
  const row = document.createElement("div");
  row.className = "feedback";
  row.innerHTML = `<span class="muted">Was this helpful?</span>
    <button class="btn ghost small" data-r="up" aria-label="Helpful">👍</button>
    <button class="btn ghost small" data-r="down" aria-label="Not helpful">👎</button>`;
  const send = async (rating, comment = "") => {
    try {
      await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating, question, comment, scenario, kind: scenario ? "guide" : "passages" }) });
      row.innerHTML = `<span class="muted">Thanks, that helps.</span>`;
    } catch { row.innerHTML = `<span class="muted">Couldn't send that just now.</span>`; }
  };
  row.addEventListener("click", (e) => {
    const r = e.target.closest("[data-r]")?.dataset.r;
    if (r === "up") return send("up");
    if (r !== "down") return;
    row.innerHTML = `<textarea rows="2" placeholder="What was wrong or missing? (optional; don't include names or addresses)"></textarea>
      <button class="btn ghost small" type="button">Send</button>`;
    $("button", row).onclick = () => send("down", $("textarea", row).value.trim());
    $("textarea", row).focus();
  });
  block.append(row);
}

function showCheck(block, check) {
  const note = $(".check-note", block);
  const text = $(".answer-text", block);
  const offTopic = check.issues.filter((i) => i.kind === "relevance");
  const problems = check.issues.filter((i) => i.kind !== "relevance");

  // Passages Jev judged off-topic are hidden, not shown with a warning: a correct quote that
  // doesn't answer the question is noise.
  const hidden = [];
  for (const i of offTopic) {
    const ptr = /\{[\d.\-]+\}/.exec(i.detail)?.[0];
    text.querySelectorAll(`.pq[data-ptr="${ptr}"]`).forEach((el) => {
      const line = el.closest("p, li") ?? el;
      line.classList.add("off-topic-line");
      line.hidden = true;
      hidden.push(line);
    });
  }
  const visibleRules = [...text.querySelectorAll(".pq")].filter((el) => {
    const src = el.closest("p, li");
    return !(src?.hidden) && !/LTB decision/.test(el.querySelector(".q-src")?.textContent ?? "") && !src?.textContent.startsWith("In [");
  });
  if (hidden.length && !visibleRules.length) {
    text.insertAdjacentHTML("afterbegin", `<p class="not-found"><strong>I couldn't find the part of Ontario law that answers this directly.</strong>
      The passages search turned up don't fit your question. Try rewording it, or check with the LTB (1-888-332-3234) or a community legal clinic.</p>`);
  }
  if (hidden.length) {
    const btn = document.createElement("button");
    btn.className = "btn ghost small show-hidden";
    btn.textContent = `Show ${hidden.length} passage${hidden.length > 1 ? "s" : ""} that may not apply`;
    btn.onclick = () => { hidden.forEach((l) => (l.hidden = false)); btn.remove(); };
    text.after(btn);
  }

  note.hidden = false;
  const counted = [check.quotes && `${check.quotes} quote${check.quotes > 1 ? "s" : ""}`,
                   check.facts && `${check.facts} fact${check.facts > 1 ? "s" : ""}`].filter(Boolean).join(" and ");
  if (!problems.length) {
    note.className = "check-note ok";
    note.textContent = counted ? `Every quote is the source's exact wording` + (hidden.length ? ` · ${hidden.length} less relevant hidden` : "") : "";
    note.hidden = !counted;
    return;
  }
  note.className = "check-note warn";
  const plural = problems.length > 1;
  note.innerHTML = `<strong>${problems.length} part${plural ? "s" : ""} couldn't be verified against the sources.</strong> ` +
    `Treat ${plural ? "them" : "it"} with caution:<ul>` +
    problems.map((i) => `<li>${esc(i.detail)}: <em>${esc(i.text.slice(0, 140))}</em></li>`).join("") + "</ul>";
  for (const q of block.querySelectorAll(".answer-text q")) {
    if (problems.some((i) => i.kind === "quote" && q.textContent.startsWith(i.text.slice(0, 60)))) q.classList.add("unverified");
  }
}

// ---------- source drawer ----------
let lastFocus = null;
function openSource(src, quotes) {
  if (!src) return;
  lastFocus = document.activeElement;
  $("#drawer-kind").className = "kind " + src.kind;
  $("#drawer-kind").textContent = KIND_NAME[src.kind];
  $("#drawer-title").textContent = src.label;
  let html = esc(src.text);
  let marked = 0;
  for (const q of quotes) {
    const needle = esc(q.trim());
    if (needle.length > 15 && html.includes(needle)) { html = html.replace(needle, `<mark>${needle}</mark>`); marked++; }
  }
  $("#drawer-body").innerHTML = html;
  const credit = {
    case: "Names are removed. Source: LTB Order Catalogue, Open Government Licence – Ontario.",
    rta: "© King's Printer for Ontario. Not an official version.",
    lease: "Government of Ontario, Residential Tenancy Agreement (Standard Form of Lease), form 2229E.",
    guide: "Landlord and Tenant Board, Tribunals Ontario.",
    mine: "From the lease you shared in this conversation (names and contact details removed).",
  }[src.kind];
  $("#drawer-note").textContent = (marked ? "Highlighted: the passage the answer quotes. " : "") + credit;
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#scrim").hidden = false;
  $("#drawer-body mark")?.scrollIntoView({ block: "center" });
  $("#drawer-close").focus();
}
function closeSource() {
  if (!$("#drawer").classList.contains("open")) return;
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#scrim").hidden = true;
  lastFocus?.focus();
}
$("#drawer-close").addEventListener("click", closeSource);
$("#scrim").addEventListener("click", closeSource);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSource(); });

// ---------- chat ----------
const chat = { turns: [], busy: false, abort: null, lease: [], leaseFacts: {} };

// A long paste that reads like a lease is kept for this conversation instead of being answered as a question.
const looksLikeLease = (t) =>
  (t.length > 800 && /standard form of lease|tenancy agreement|lease agreement/i.test(t)) ||
  (t.length > 1500 && (t.match(/\b(landlord|tenant|rent)\b/gi) ?? []).length > 15);

function setLease(terms, facts = {}) {
  chat.lease = terms;
  chat.leaseFacts = terms.length ? facts : {};
  $("#lease-pill").hidden = !terms.length;
  $("#lease-pill-text").textContent = `Using your lease in answers (${terms.length} terms)`;
}
$("#lease-remove").addEventListener("click", () => setLease([]));

async function shareLease(text) {
  $("#empty").hidden = true;
  $("#new-chat").hidden = false;
  const thread = $("#thread");
  const mine = bubble(`Shared a lease (${Math.round(text.length / 1000)}k characters)`);
  mine.classList.add("lease-shared");
  thread.append(mine);
  const slot = document.createElement("div");
  slot.className = "msg bot";
  slot.innerHTML = `<p class="thinking">Reading your lease…</p>`;
  thread.append(slot);
  try {
    const res = await fetch("/api/lease", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    setLease(data.terms, data.facts ?? {});
    slot.innerHTML = `<div class="answer lease-ack"><p>I've read your lease: <strong>${data.terms.length} terms</strong>. Ask me anything about your tenancy,
      and each answer will show what your lease says next to Ontario law, and flag terms that may conflict with it.</p>
      <p class="muted">Names, emails, phone numbers and addresses are removed. The lease is only kept for this conversation.</p>
      <button class="btn ghost small" type="button">Check every term of this lease</button></div>`;
    $("button", slot).onclick = () => {
      const form = new FormData();
      form.append("text", text);
      document.querySelector('.tab[aria-controls="view-review"]')?.click();
      startReview(form);
    };
  } catch (err) {
    console.warn("lease failed:", err);
    slot.innerHTML = `<p class="muted">I couldn't read that as a lease. You can also upload it under “Review a lease”.</p>`;
  }
}
const role = () => document.querySelector("input[name=role]:checked")?.value || null;

function setRoleHint() {
  const r = role();
  $("#role-hint").textContent = r
    ? `Answers stay neutral. Ask for help with a negotiation and you'll get suggestions for the ${r}'s side.`
    : "Answers are neutral either way.";
}
for (const r of document.querySelectorAll("input[name=role]")) r.addEventListener("change", setRoleHint);

$("#starters").innerHTML = STARTERS.map((s, i) =>
  `<button type="button" class="starter" data-i="${i}">${esc(s.text)}</button>`).join("");
$("#starters").addEventListener("click", (e) => {
  const b = e.target.closest(".starter");
  if (!b) return;
  const s = STARTERS[b.dataset.i];
  if (s.needsRole && !role()) {
    document.querySelector("input[name=role][value=tenant]").checked = true;
    setRoleHint();
  }
  ask(s.clause ?? s.text, Boolean(s.clause), s.text);
});

function bubble(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  return el;
}

async function ask(text, isClause, display = text) {
  if (chat.busy || !text.trim()) return;
  if (!isClause && looksLikeLease(text)) return shareLease(text);
  chat.busy = true;
  $("#empty").hidden = true;
  $("#new-chat").hidden = false;
  $("#ask-go").disabled = true;
  const thread = $("#thread");
  thread.append(bubble(display));
  const slot = document.createElement("div");
  slot.className = "msg bot";
  thread.append(slot);
  slot.scrollIntoView({ behavior: "smooth", block: "start" });

  chat.abort = new AbortController();
  const payload = {
    text, mode: isClause ? "clause" : "question", role: role(),
    history: isClause ? [] : chat.turns.slice(-2),
    lease: isClause ? [] : chat.lease,
    lease_facts: isClause ? {} : chat.leaseFacts,
  };
  const { state, answer, scenario } = await runAnswer(slot, payload, chat.abort.signal);
  if (state === "done") chat.turns.push({ question: isClause ? `A lease says: "${text}"` : text, answer, scenario });
  chat.busy = false;
  $("#ask-go").disabled = false;
  $("#ask-text").focus();
  loadStatus();
}

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#ask-text").value.trim();
  if (!text) return $("#ask-text").focus();
  $("#ask-text").value = "";
  const isClause = $("#is-clause").checked;
  $("#is-clause").checked = false;
  ask(text, isClause);
});
$("#ask-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});
$("#new-chat").addEventListener("click", () => {
  chat.abort?.abort();
  chat.turns = [];
  chat.busy = false;
  setLease([]);
  $("#thread").replaceChildren($("#empty"));
  $("#empty").hidden = false;
  $("#new-chat").hidden = true;
  $("#ask-go").disabled = false;
});

// ---------- review ----------
const review = { clauses: [], queue: [], running: false, paused: false, abort: null, done: 0, total: 0, timings: [] };
const CHECK_NAMES = {
  no_pets: "No-pets rule", extra_deposit: "Deposit not allowed", extra_fees: "Extra fee", rent_increase_rules: "Rent increase rules",
  entry_without_notice: "Entry without notice", early_exit_penalty: "Early-exit penalty", repairs_shifted: "Repairs shifted",
  post_dated_cheques: "Required post-dated cheques", must_move_out_at_end: "Move-out at term end", waives_the_act: "Waives the Act",
  vital_services: "Vital services", lock_changes: "Lock changes", guest_limits: "Guest limits", late_fees: "Late fees", sublet_ban: "Sublet ban",
  auto_renewal: "Automatic renewal", liability_waiver: "Liability waiver", no_rent_abatement: "No rent reduction", legal_costs: "Legal costs",
};

async function startReview(form) {
  $("#review-start").hidden = true;
  $("#review-run").hidden = false;
  $("#run-title").textContent = "Reading the lease…";
  $("#run-sub").textContent = "";
  $("#clauses").replaceChildren();
  let data;
  try {
    const res = await fetch("/api/split", { method: "POST", body: form });
    data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
  } catch (err) {
    $("#run-title").textContent = "Couldn't read that lease";
    $("#run-sub").textContent = err.message;
    return;
  }

  review.clauses = data.clauses.map((c) => ({ ...c, el: clauseCard(c) }));
  review.terms = data.clauses.filter((c) => !c.skip).map(({ heading, text }) => ({ heading, text }));
  review.facts = data.facts ?? {};
  $("#clauses").append(...review.clauses.map((c) => c.el));

  let toCheck = review.clauses.filter((c) => !c.skip);
  await loadStatus();
  if (status.jev) {
    $("#run-title").textContent = "Scanning terms with Jev…";
    try {
      const res = await fetch("/api/scan", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clauses: toCheck.map(({ index, heading, text }) => ({ index, heading, text })) }) });
      const { findings } = await res.json();
      const byIndex = Object.fromEntries(findings.map((f) => [f.index, f]));
      for (const c of toCheck) showFlags(c, byIndex[c.index]);
      const rank = (c) => Math.min(...(byIndex[c.index]?.flags ?? []).map((f) => ({ high: 0, medium: 1, low: 2 })[f.severity]), 3);
      // Explain only what the scan flagged; the rest can still be checked one at a time.
      const flagged = toCheck.filter((c) => byIndex[c.index]?.flags.length);
      for (const c of toCheck.filter((c) => !byIndex[c.index]?.flags.length)) {
        c.el.dataset.state = "queued";
        $(".badge", c.el).textContent = "No issues found in the scan";
        c.el.classList.add("scanned-ok");
      }
      toCheck = flagged.sort((a, b) => rank(a) - rank(b));
    } catch { /* fall back to checking every clause in order */ }
  }

  review.queue = toCheck;
  review.total = toCheck.length;
  review.done = 0;
  review.timings = [];
  review.paused = false;
  $("#run-stop").textContent = "Pause";
  updateRunBar();
  pump();
}

function clauseCard(c) {
  const el = $("#clause-tpl").content.firstElementChild.cloneNode(true);
  $(".clause-num", el).textContent = c.index + 1;
  // "1. TERM. The lease begins…" → title "1. TERM.", body "The lease begins…"
  const short = /^\s*((?:\d{1,2}(?:\.\d+)*[.)]|(?:[Ss]ection|SECTION|[Aa]rticle|ARTICLE) [\dIVXLC]+[.:]?)?\s*[A-Z][A-Z &/'’-]{2,}[.:]?)(?=\s|$)/.exec(c.text);
  const title = short && short[1].length < 60 ? short[1].trim() : c.heading;
  $(".clause-title", el).textContent = title;
  $(".clause-text", el).textContent = c.text.startsWith(title) ? c.text.slice(title.length).trim() || c.text : c.text;
  el.dataset.state = c.skip ? "skipped" : "queued";
  $(".badge", el).textContent = c.form ? "Form details (not checked)" : c.skip ? "Heading" : "Waiting";
  $(".check-now", el).addEventListener("click", () => {
    el.classList.remove("scanned-ok");
    review.total += review.queue.includes(c) ? 0 : 1;
    review.queue = [c, ...review.queue.filter((q) => q !== c)];
    if (review.paused) togglePause();
    else pump();
  });
  return el;
}

function showFlags(c, finding) {
  if (!finding?.flags.length) return;
  $(".flags", c.el).innerHTML = finding.flags.map((f) =>
    `<span class="flag ${f.severity}" title="${esc(f.question)}">${esc(CHECK_NAMES[f.key] ?? f.key)} · ${esc(f.law)}</span>`).join("");
}

function updateRunBar() {
  const left = review.queue.length;
  $("#run-progress").style.width = review.total ? `${(review.done / review.total) * 100}%` : "0";
  if (!left && !review.running) {
    $("#run-title").textContent = "Review finished";
    const scanned = review.clauses.filter((c) => c.el.classList.contains("scanned-ok")).length;
    $("#run-sub").textContent = `${review.done} flagged term${review.done === 1 ? "" : "s"} explained` +
      (scanned ? ` · ${scanned} more scanned with no issues found (open any to check it)` : "") + ".";
    $("#run-stop").hidden = true;
    return;
  }
  $("#run-stop").hidden = false;
  $("#run-title").textContent = review.paused ? "Paused" : "Reviewing the lease";
  // Estimate from how long terms have actually taken (median), not a fixed guess.
  const t = [...review.timings].sort((a, b) => a - b);
  const per = t.length ? t[Math.floor(t.length / 2)] : (status.jev ? 2 : 25);
  const secs = Math.round(left * per);
  const eta = secs < 60 ? "under a minute left" : `about ${Math.ceil(secs / 60)} min left`;
  $("#run-sub").textContent = `${review.done} of ${review.total} terms checked · ${eta}`;
}

async function pump() {
  if (review.running || review.paused) return;
  const c = review.queue.shift();
  if (!c) return updateRunBar();
  review.running = true;
  c.el.dataset.state = "working";
  $(".badge", c.el).textContent = "Reading the law";
  updateRunBar();
  review.abort = new AbortController();
  const started = performance.now();
  let { state } = await runAnswer($(".answer-slot", c.el), { text: c.text, mode: "clause" }, review.abort.signal);
  if (state === "error" && !review.paused) {  // one quiet retry before giving up
    ({ state } = await runAnswer($(".answer-slot", c.el), { text: c.text, mode: "clause" }, review.abort.signal));
  }
  review.timings.push((performance.now() - started) / 1000);
  review.running = false;
  if (state === "aborted") {
    review.queue.unshift(c);
    c.el.dataset.state = "queued";
    $(".badge", c.el).textContent = "Waiting";
    $(".answer-slot", c.el).replaceChildren();
  } else if (state === "done") {
    c.el.dataset.state = "done";
    $(".badge", c.el).textContent = "Checked";
    review.done++;
  } else {
    c.el.dataset.state = "queued";
    $(".badge", c.el).textContent = "Not checked yet";
    $(".answer-slot", c.el).innerHTML = `<p class="muted">Couldn't check this one just now.</p>`;
    $(".check-now", c.el).textContent = "Retry";
    review.done++;
  }
  updateRunBar();
  pump();
}

function togglePause() {
  review.paused = !review.paused;
  $("#run-stop").textContent = review.paused ? "Resume" : "Pause";
  if (review.paused) review.abort?.abort();
  else pump();
  updateRunBar();
}
$("#run-stop").addEventListener("click", togglePause);
$("#run-ask").addEventListener("click", () => {
  setLease(review.terms ?? [], review.facts ?? {});
  document.querySelector('.tab[aria-controls="view-ask"]')?.click();
  $("#ask-text").focus();
});
$("#run-new").addEventListener("click", () => {
  review.paused = true;
  review.abort?.abort();
  review.queue = [];
  $("#review-run").hidden = true;
  $("#review-start").hidden = false;
  $("#run-stop").hidden = false;
});

const fileInput = $("#file");
fileInput.addEventListener("change", () => {
  if (!fileInput.files[0]) return;
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  fileInput.value = "";
  startReview(form);
});
const dz = $("#dropzone");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("over");
  const f = e.dataTransfer.files[0];
  if (!f) return;
  const form = new FormData();
  form.append("file", f);
  startReview(form);
});
$("#paste-toggle").addEventListener("click", () => { $("#paste-box").hidden = !$("#paste-box").hidden; $("#paste-text").focus(); });
$("#paste-go").addEventListener("click", () => {
  const text = $("#paste-text").value.trim();
  if (!text) return $("#paste-text").focus();
  const form = new FormData();
  form.append("text", text);
  startReview(form);
});
$("#try-sample").addEventListener("click", () => {
  const form = new FormData();
  form.append("sample", "true");
  startReview(form);
});

setRoleHint();
loadStatus();
