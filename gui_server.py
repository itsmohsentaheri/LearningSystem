#!/usr/bin/env python3
"""Browser GUI for the adaptive tutor (Flask).

Design: a worker thread runs ``runtime_adaptive_tutor.run()`` unchanged, but
with a ``WebIO`` adapter injected instead of the console one. The adapter
publishes each prompt/lesson as an event and blocks on a queue until the
browser POSTs the learner's answer — so grading, FSRS scheduling, remediation
and profile persistence reuse the exact same code paths as the CLI.

Usage:
    python gui_server.py [--graph curriculum_graph.json] [--simulate]
                         [--user-id NAME | --profile-file PATH]
                         [--port 8000] [--host 127.0.0.1]

Then open http://127.0.0.1:8000 in a browser.
"""
import argparse
import json
import queue
import re
import sys
import threading
from datetime import datetime, timezone

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError:  # pragma: no cover
    sys.stderr.write("Flask is required for the GUI: pip install flask\n")
    raise

import runtime_adaptive_tutor as tutor


class WebIO(tutor.ConsoleIO):
    """I/O adapter that routes every interactive prompt to the browser.

    Each prompt becomes an event pushed onto ``session.events``; execution
    blocks until ``session.answers`` receives the value posted by the UI.
    """

    def __init__(self, session):
        super().__init__()
        self.session = session

    def show(self, text):
        lines = [ln for ln in str(text).split("\n") if ln.strip()]
        if lines:
            self.session.emit({"event": "info", "lines": [ln.strip() for ln in lines]})

    def ask_question(self, question, simulated=False, simulated_correct=True):
        # In the GUI even "--simulate" content must be answered by a human.
        payload = {"type": question["type"], "question": question["question"],
                   "options": list(question.get("options") or []),
                   "accepted_answers": list(question.get("accepted_answers") or [])}
        if question["type"] == "free_text":
            return str(self.session.await_prompt(
                {"event": "question", "kind": "text", "question": payload}))
        while True:
            raw = self.session.await_prompt(
                {"event": "question", "kind": "choice", "question": payload})
            match = re.fullmatch(r"\s*([1-4])\s*", str(raw))
            if match:
                return int(match.group(1)) - 1
            payload = dict(payload, error="لطفاً یکی از گزینه‌های ۱ تا ۴ را انتخاب کنید.")

    def ask_retry_action(self, remediation_text):
        return self.session.await_prompt({"event": "action", "kind": "action",
                                          "prompt": "بعد از بازخورد، چه کاری انجام دهم؟",
                                          "options": ["retry", "explain", "stop"]})

    def ask_uncertain_action(self, feedback_text):
        return self.session.await_prompt({"event": "action", "kind": "action",
                                          "prompt": "پاسخ شما با اطمینان قابل تصحیح نبود.",
                                          "options": ["retry", "self_correct", "pause"]})

    def ask_self_rating(self, feedback_text):
        return self.session.await_prompt({"event": "action", "kind": "action",
                                          "prompt": "چقدر به پاسخ خود مطمئن هستید؟",
                                          "options": ["correct", "incorrect", "retry"]})

    def ask_self_rating_again(self):
        return self.session.await_prompt({"event": "action", "kind": "action",
                                          "prompt": "ارزیابی نهایی: پاسخ درست بود یا نادرست؟",
                                          "options": ["correct", "incorrect"]})

    def ask_portfolio_field(self, label, current):
        # Portfolio collection happens through collect_portfolio's form path;
        # per-field console fallback is not used by the GUI form below.
        return str(self.session.await_prompt({"event": "question", "kind": "text",
                                              "question": {"type": "free_text",
                                                           "question": label,
                                                           "options": [],
                                                           "accepted_answers": []}}))


class TutorSession:
    """One learner journey driven by tutor.run() inside a worker thread."""

    def __init__(self, graph_path, api_key=None, model=None, simulate=False,
                 user_id=None, profile_file=None, diagnostic_count=4,
                 chapter_quiz_questions=3):
        self.graph, self.nodes = tutor.load_graph(graph_path)
        self.course = self.graph.get("course_name", "Curriculum")
        self.nodes_by_id = {str(n["id"]): n for n in self.nodes}
        self.simulate = bool(simulate)
        self.args = _make_args(graph_path, api_key, model, simulate, user_id,
                               profile_file, diagnostic_count, chapter_quiz_questions)
        self.profile_file = tutor.profile_path(self.args)
        self.events = queue.Queue()          # prompts/info -> HTTP layer
        self.answers = queue.Queue()         # answers -> worker thread
        self.status = "starting"             # starting|running|awaiting|done|error|paused
        self.log_lines = []
        self.pending = None
        self.exit_code = None
        self.error = None
        self._thread = None

    # -- logging -----------------------------------------------------------
    def add_log(self, message):
        text = str(message)
        if text.startswith("[tutor] "):
            text = text[len("[tutor] "):]
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log_lines.append({"time": stamp, "text": text})
        if len(self.log_lines) > 500:
            del self.log_lines[:-500]

    # -- worker ------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        ui = WebIO(self)
        orig_log = tutor.log
        try:
            tutor.log = self.add_log
            code = tutor.run(self.args, io=ui)
            self.exit_code = code if isinstance(code, int) else 0
            self.status = "done"
            self.events.put({"event": "done", "exit_code": self.exit_code})
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            self.status = "error"
            self.error = str(exc)
            self.add_log("ERROR: " + str(exc))
            self.events.put({"event": "error", "message": str(exc)})
        finally:
            tutor.log = orig_log

    # -- prompt plumbing (called from the worker thread) ---------------------
    def emit(self, descriptor):
        self.events.put(descriptor)

    def await_prompt(self, descriptor):
        """Publish a prompt to the browser and block until its answer arrives."""
        self.pending = descriptor
        self.status = "awaiting"
        self.events.put(descriptor)
        answer = self.answers.get()
        self.pending = None
        self.status = "running"
        return answer

    # -- read-only views ------------------------------------------------------
    def progress_view(self):
        try:
            profile = tutor.load_profile(self.profile_file)
        except (OSError, ValueError, json.JSONDecodeError):
            profile = {}
        mastered = set(profile.get("mastered_topics", [])) & set(self.nodes_by_id)
        scheduler = tutor.SpacedRepetitionEngine(dict(profile), None, self.nodes)
        due = scheduler.get_due_reviews()
        chapters = []
        for name, nodes in tutor._chapter_groups(self.nodes).items():
            done = sum(1 for n in nodes if n["id"] in mastered)
            quiz = (profile.get("chapter_quizzes") or {}).get(name) or {}
            chapters.append({"name": name, "total": len(nodes), "done": done,
                             "quiz_score": quiz.get("score"),
                             "quiz_done": bool(quiz.get("completed")),
                             "weak": bool(quiz.get("weak"))})
        reviews = []
        for topic_id, state in sorted((profile.get("fsrs_reviews") or {}).items()):
            node = self.nodes_by_id.get(topic_id, {})
            reviews.append({"topic_id": topic_id, "title": node.get("title", topic_id),
                            "due_date": state.get("due_date"), "reps": state.get("reps", 0),
                            "interval_days": state.get("interval_days"),
                            "stability": state.get("stability"),
                            "difficulty": state.get("difficulty"),
                            "last_rating": state.get("last_rating")})
        events = [e for e in profile.get("history", []) if isinstance(e, dict)]
        counts = {}
        for e in events:
            counts[e.get("event", "?")] = counts.get(e.get("event", "?"), 0) + 1
        return {
            "course": self.course,
            "topics_total": len(self.nodes),
            "topics_mastered": len(mastered),
            "status": self.status,
            "mode": "simulate" if self.simulate else "live",
            "chapters": sorted(chapters, key=lambda c: c["name"]),
            "due_reviews": due,
            "fsrs_reviews": reviews,
            "portfolio": profile.get("portfolio", {}),
            "diagnostic_completed": bool((profile.get("diagnostic_state") or {}).get("completed")),
            "events_summary": counts,
            "recent_history": events[-12:][::-1],
            "profile_file": self.profile_file,
        }


def _make_args(graph_path, api_key, model, simulate, user_id, profile_file,
               diagnostic_count, chapter_quiz_questions):
    ns = argparse.Namespace()
    ns.graph = graph_path
    ns.api_key = api_key
    ns.model = model
    ns.simulate = bool(simulate)
    ns.user_id = user_id
    ns.profile_file = profile_file
    ns.check_due_reviews = False
    ns.update_portfolio = False
    ns.diagnostic_question_count = diagnostic_count
    ns.chapter_quiz_questions = chapter_quiz_questions
    return ns


# ---------------------------------------------------------------------------
# Portfolio form: replace input()-based collection with a single web form.
# ---------------------------------------------------------------------------
_ORIG_COLLECT_PORTFOLIO = tutor.collect_portfolio


def _gui_collect_portfolio(profile, path, force=False, simulated=False, io=None):
    if not isinstance(io, WebIO):
        return _ORIG_COLLECT_PORTFOLIO(profile, path, force, simulated, io=io)
    session = io.session
    fields = tutor.PORTFOLIO_FIELDS
    portfolio = profile.setdefault("portfolio", {})
    if not isinstance(portfolio, dict):
        portfolio = profile["portfolio"] = {}
    missing = not all(str(portfolio.get(key, "")).strip() for key, _ in fields)
    if not force and not missing:
        return portfolio
    if simulated:
        # Offline mock keeps the CLI defaults; no prompt needed.
        return _ORIG_COLLECT_PORTFOLIO(profile, path, force, simulated, io=tutor.ConsoleIO())
    descriptors = [{"key": key, "label": label, "value": str(portfolio.get(key, ""))}
                   for key, label in fields]
    values = session.await_prompt({"event": "form", "kind": "form",
                                   "title": "پروفایل یادگیرنده — برای شخصی‌سازی دوره",
                                   "fields": descriptors})
    if not isinstance(values, dict):
        values = {}
    for key, _label in fields:
        value = str(values.get(key, "")).strip()
        if value:
            portfolio[key] = value
    profile["portfolio"] = portfolio
    tutor.save_profile(path, profile)
    return portfolio


tutor.collect_portfolio = _gui_collect_portfolio


# ---------------------------------------------------------------------------
# Flask application (single active session model)
# ---------------------------------------------------------------------------
app = Flask(__name__)
STATE = {"session": None, "lock": threading.Lock()}
CONFIG = {"graph_path": "curriculum_graph.json", "api_key": None, "model": None,
          "simulate": False, "user_id": None, "profile_file": None,
          "diagnostic_count": 4, "chapter_quiz_questions": 3}


def _require_session():
    session = STATE["session"]
    if session is None:
        raise ValueError("No active session. Start one from the page first.")
    return session


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE)


@app.route("/api/start", methods=["POST"])
def api_start():
    body = request.get_json(force=True, silent=True) or {}
    config = dict(CONFIG)
    for key in ("user_id", "profile_file"):
        if body.get(key):
            config[key] = str(body[key]).strip()
    if "simulate" in body:
        config["simulate"] = bool(body["simulate"])
    try:
        session = TutorSession(**config)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return jsonify({"event": "error", "message": str(exc)}), 400
    with STATE["lock"]:
        old = STATE["session"]
        STATE["session"] = session
    if old is not None and old.status in ("starting", "running", "awaiting"):
        old.status = "paused"  # let abandoned worker unblock harmlessly
    session.start()
    return jsonify({"event": "started", "course": session.course,
                    "topics": len(session.nodes),
                    "mode": "simulate" if session.simulate else "live"})


@app.route("/api/next", methods=["POST"])
def api_next():
    timeout = float(request.args.get("timeout", 25))
    try:
        session = _require_session()
    except ValueError as exc:
        return jsonify({"event": "error", "message": str(exc)}), 400
    try:
        event = dict(session.events.get(timeout=timeout))
    except queue.Empty:
        return jsonify({"event": "heartbeat", "status": session.status})
    event["logs"] = list(session.log_lines[-8:])
    return jsonify(event)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    body = request.get_json(force=True, silent=True) or {}
    try:
        session = _require_session()
    except ValueError as exc:
        return jsonify({"event": "error", "message": str(exc)}), 400
    kind = (body.get("kind") or "").strip()
    raw = body.get("value", "")
    if kind == "form":
        session.answers.put(raw if isinstance(raw, dict) else {})
    elif kind == "action":
        action = str(raw).strip().lower()
        mapping = {"retry": "r", "explain": "e", "stop": "q", "pause": "n",
                   "self_correct": "s", "correct": "1", "incorrect": "2"}
        session.answers.put(mapping.get(action, action[:1] or ""))
    elif kind == "choice":
        match = re.fullmatch(r"\s*([1-4])\s*", str(raw))
        if not match:
            return jsonify({"event": "rejected", "message": "Select one of the four options."})
        session.answers.put(match.group(1))
    else:
        session.answers.put(str(raw))
    return jsonify({"event": "accepted"})


@app.route("/api/status")
def api_status():
    session = STATE["session"]
    if session is None:
        return jsonify({"active": False})
    payload = session.progress_view()
    payload["active"] = True
    payload["logs"] = session.log_lines[-40:]
    payload["pending_event"] = session.pending
    return jsonify(payload)


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>معلم تطبیقی — رابط گرافیکی</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --ink:#e2e8f0; --muted:#94a3b8;
          --accent:#38bdf8; --ok:#34d399; --bad:#f87171; --warn:#fbbf24; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: Tahoma, 'Segoe UI', sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:14px 20px; background:#0b1220; display:flex; gap:12px;
           align-items:center; flex-wrap:wrap; border-bottom:1px solid #24314a; }
  header h1 { font-size:17px; margin:0; color:var(--accent); }
  .badge { font-size:12px; padding:2px 10px; border-radius:999px; background:#24314a; }
  main { display:grid; grid-template-columns: 2fr 1fr; gap:16px; padding:16px 20px; }
  @media (max-width: 900px){ main{ grid-template-columns:1fr; } }
  .card { background:var(--card); border-radius:12px; padding:16px; margin-bottom:16px; border:1px solid #2b3a55; }
  .card h2 { margin:0 0 10px; font-size:15px; color:var(--accent); }
  button { cursor:pointer; border:none; border-radius:8px; padding:9px 16px; font-size:14px;
           background:var(--accent); color:#082f49; font-weight:bold; }
  button.secondary { background:#334155; color:var(--ink); }
  button:hover { filter:brightness(1.1); }
  input[type=text] { width:100%; padding:9px; border-radius:8px; border:1px solid #334155;
                     background:#0b1220; color:var(--ink); font-size:14px; }
  .opt { display:block; width:100%; text-align:right; margin:6px 0; background:#223049;
         color:var(--ink); font-weight:normal; padding:10px 14px; border-radius:8px; border:1px solid #334155; }
  .qtext { font-size:15px; line-height:1.9; white-space:pre-wrap; }
  .feedback { margin-top:10px; padding:10px 12px; border-radius:8px; font-size:13px;
              background:#0b1220; border:1px solid #334155; white-space:pre-wrap; }
  .progress-track { height:10px; background:#0b1220; border-radius:999px; overflow:hidden; }
  .progress-fill { height:100%; background:linear-gradient(90deg,#38bdf8,#34d399); width:0%; transition:width .4s; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th, td { padding:6px 8px; border-bottom:1px solid #24314a; text-align:right; }
  th { color:var(--muted); font-weight:normal; }
  .log { font-size:12px; max-height:240px; overflow:auto; direction:ltr; text-align:left; }
  .log .t { color:var(--muted); margin-left:8px; }
  .muted { color:var(--muted); font-size:12.5px; }
  .pill-ok{color:var(--ok)} .pill-bad{color:var(--bad)} .pill-warn{color:var(--warn)}
  #toast { position:fixed; bottom:18px; left:18px; background:#223049; padding:10px 16px;
           border-radius:10px; border:1px solid #334155; display:none; font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>معلم تطبیقی — رابط گرافیکی</h1>
  <span class="badge" id="courseBadge">—</span>
  <span class="badge" id="modeBadge">آماده شروع</span>
  <button onclick="startSession()">شروع / ادامه جلسه</button>
  <button class="secondary" onclick="loadStatus()">وضعیت دوره</button>
</header>
<main>
  <section>
    <div class="card">
      <h2>تعامل</h2>
      <div id="interaction" class="muted">برای شروع روی «شروع / ادامه جلسه» کلیک کنید.</div>
    </div>
    <div class="card">
      <h2>پیشرفت دوره</h2>
      <div class="progress-track"><div class="progress-fill" id="progFill"></div></div>
      <div class="muted" id="progText" style="margin-top:6px">—</div>
      <table id="chapterTable"><thead><tr><th>فصل</th><th>تکمیل</th><th>آزمون فصل</th></tr></thead><tbody></tbody></table>
    </div>
  </section>
  <aside>
    <div class="card">
      <h2>مرورهای FSRS (تکرار فاصله‌دار)</h2>
      <div id="fsrsBox" class="muted">هنوز داده‌ای ثبت نشده است.</div>
    </div>
    <div class="card">
      <h2>لاگ زنده</h2>
      <div class="log" id="logBox"></div>
    </div>
  </aside>
</main>
<div id="toast"></div>
<script>
let polling = false;
const $ = id => document.getElementById(id);
function toast(msg){ const t=$('toast'); t.textContent=msg; t.style.display='block';
  setTimeout(()=>t.style.display='none', 2600); }
async function post(url, body){
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                                body: JSON.stringify(body||{})});
  return res.json(); }
function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

async function startSession(){
  const cfg = {};
  const uid = prompt('شناسه کاربری (اختیاری — پروفایل جدا برای هر نفر):','');
  if(uid && uid.trim()) cfg.user_id = uid.trim();
  cfg.simulate = confirm('حالت شبیه‌سازی آفلاین؟\nOK = آفلاین (بدون API) — Cancel = اتصال به OpenRouter');
  const r = await post('/api/start', cfg);
  if(r.event==='error'){ toast('خطا: '+r.message); return; }
  $('courseBadge').textContent = r.course + ' — ' + r.topics + ' موضوع';
  $('modeBadge').textContent = r.mode==='simulate' ? 'شبیه‌سازی آفلاین' : 'لایو (OpenRouter)';
  pollLoop();
}

async function pollLoop(){
  if(polling) return; polling = true;
  while(true){
    let ev;
    try { ev = await (await fetch('/api/next?timeout=25',{method:'POST'})).json(); }
    catch(e){ await new Promise(s=>setTimeout(s,1500)); continue; }
    if(ev.event==='heartbeat'){ loadStatus(true);
      if(ev.status==='done'||ev.status==='error'||ev.status==='paused') break; continue; }
    renderEvent(ev);
    if(ev.event==='done' || ev.event==='error') break;
  }
  polling = false;
}

function appendLogs(lines){ (lines||[]).forEach(l=>{
  const box=$('logBox'); const div=document.createElement('div');
  div.innerHTML='<span class="t">'+esc(l.time)+'</span>'+esc(l.text);
  box.appendChild(div); box.scrollTop=box.scrollHeight; }); }

function renderEvent(ev){
  appendLogs(ev.logs);
  const el = $('interaction');
  if(ev.event==='info'){ el.innerHTML = '<div class="qtext">'+ev.lines.map(esc).join('<br>')+'</div>'; return; }
  if(ev.event==='question'){
    const q = ev.question;
    let html = '<div class="qtext">'+esc(q.question)+'</div>';
    if(q.error) html += '<div class="feedback pill-bad">'+esc(q.error)+'</div>';
    if(q.type==='free_text'){
      html += '<div style="margin-top:12px"><input type="text" id="ansText" placeholder="پاسخ کوتاه…"'+
              ' onkeydown="if(event.key===\'Enter\')sendText()"></div>'+
              '<button style="margin-top:10px" onclick="sendText()">ارسال پاسخ</button>';
    } else {
      html += '<div style="margin-top:12px">';
      q.options.forEach((o,i)=>{ html += '<button class="opt" onclick="sendChoice('+(i+1)+')">'+(i+1)+'. '+esc(o)+'</button>'; });
      html += '</div>';
    }
    el.innerHTML = html;
    const inp=$('ansText'); if(inp) inp.focus();
    return;
  }
  if(ev.event==='action'){
    const labels={retry:'تلاش دوباره',explain:'توضیح بیشتر',stop:'توقف جلسه',pause:'توقف (مطمئن نیستم)',
                  self_correct:'مطمئنم درست بود',correct:'درست بود',incorrect:'نادرست بود'};
    let html = '<div class="qtext">'+esc(ev.prompt)+'</div><div style="margin-top:12px">';
    ev.options.forEach(o=>{ html+='<button class="opt" onclick="sendAction(\''+o+'\')">'+(labels[o]||o)+'</button>'; });
    el.innerHTML = html + '</div>';
    return;
  }
  if(ev.event==='form'){
    let html = '<div class="qtext">'+esc(ev.title)+'</div>';
    ev.fields.forEach(f=>{ html += '<div style="margin-top:10px"><label class="muted">'+esc(f.label)+
      '</label><input type="text" id="fld_'+f.key+'" value="'+esc(f.value)+'"></div>'; });
    html += '<button style="margin-top:14px" onclick="sendForm()">ذخیره و ادامه</button>';
    el.innerHTML = html;
    return;
  }
  if(ev.event==='done'){ el.innerHTML = '<div class="qtext pill-ok">✅ جلسه پایان یافت (کد خروج '+ev.exit_code+').</div>'; loadStatus(); return; }
  if(ev.event==='error'){ el.innerHTML = '<div class="qtext pill-bad">⛔ '+esc(ev.message)+'</div>'; loadStatus(); return; }
}

async function sendChoice(n){ await post('/api/answer',{kind:'choice', value:String(n)}); }
async function sendText(){ const v=$('ansText').value; if(!v.trim()){ toast('پاسخ خالی نیست؛ دوباره تلاش کنید.'); return; }
  await post('/api/answer',{kind:'text', value:v}); }
async function sendAction(a){ await post('/api/answer',{kind:'action', value:a}); }
async function sendForm(){ const vals={};
  ['name','goal','background','preferences'].forEach(k=>{ const e=$('fld_'+k); if(e) vals[k]=e.value; });
  await post('/api/answer',{kind:'form', value:vals}); }

async function loadStatus(light){
  try{
    const s = await (await fetch('/api/status')).json();
    if(!s.active) return;
    const pct = s.topics_total? Math.round(100*s.topics_mastered/s.topics_total):0;
    $('progFill').style.width = pct+'%';
    $('progText').textContent = s.topics_mastered+' از '+s.topics_total+' موضوع ('+pct+'%) — وضعیت: '+s.status;
    const tb = document.querySelector('#chapterTable tbody'); tb.innerHTML='';
    s.chapters.forEach(c=>{ const tr=document.createElement('tr');
      const quiz = c.quiz_done ? ((Math.round((c.quiz_score||0)*100))+'%'+(c.weak?' ⚠️':' ✅')) : '—';
      tr.innerHTML='<td>'+esc(c.name)+'</td><td>'+c.done+'/'+c.total+'</td><td>'+quiz+'</td>';
      tb.appendChild(tr); });
    const fb = $('fsrsBox');
    if(s.fsrs_reviews && s.fsrs_reviews.length){
      let html='<table><thead><tr><th>موضوع</th><th>مرور بعدی</th><th>فاصله</th><th>تعداد مرور</th></tr></thead><tbody>';
      s.fsrs_reviews.forEach(r=>{ const overdue = r.due_date && new Date(r.due_date)<=new Date();
        html+='<tr><td>'+esc(r.title)+'</td><td class="'+(overdue?'pill-warn':'')+'">'+
              esc((r.due_date||'').slice(0,16).replace('T',' '))+(overdue?' (معوق)':'')+'</td><td>'+
              (r.interval_days!=null? r.interval_days+' روز':'—')+'</td><td>'+r.reps+'</td></tr>'; });
      fb.innerHTML = html+'</tbody></table>';
    } else fb.textContent='هنوز داده‌ای ثبت نشده است.';
    if(!light) appendLogs(s.logs.slice(-8));
  }catch(e){ /* transient status errors are ignored */ }
}
setInterval(()=>{ if(polling) loadStatus(true); }, 8000);
</script>
</body>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Browser GUI for the adaptive tutor.")
    parser.add_argument("--graph", default="curriculum_graph.json", help="Path to curriculum graph JSON.")
    parser.add_argument("--api-key", "-k", default=None, help="OpenRouter API key; otherwise OPENROUTER_API_KEY is used.")
    parser.add_argument("--model", default=None, help="OpenRouter model identifier (or OPENROUTER_MODEL in .env).")
    parser.add_argument("--simulate", action="store_true", help="Default new sessions to offline mock content.")
    parser.add_argument("--user-id", default=None, help="Default learner identifier for saved profiles.")
    parser.add_argument("--profile-file", default=None, help="Explicit learner JSON profile path.")
    parser.add_argument("--diagnostic-question-count", type=int, default=4)
    parser.add_argument("--chapter-quiz-questions", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default localhost only).")
    parser.add_argument("--port", type=int, default=8000)
    args, _unknown = parser.parse_known_args(argv)

    # Defaults are applied at import time so the app is usable (e.g. via tests)
    # without going through main(); CLI flags override them here.
    CONFIG.update({
        "graph_path": args.graph,
        "api_key": args.api_key if args.api_key is not None else CONFIG.get("api_key"),
        "model": args.model if args.model is not None else CONFIG.get("model"),
        "simulate": args.simulate or CONFIG.get("simulate", False),
        "user_id": args.user_id or CONFIG.get("user_id"),
        "profile_file": args.profile_file or CONFIG.get("profile_file"),
        "diagnostic_count": args.diagnostic_question_count,
        "chapter_quiz_questions": args.chapter_quiz_questions,
    })
    print("Adaptive Tutor GUI -> http://%s:%d  (graph: %s)" % (args.host, args.port, args.graph))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
