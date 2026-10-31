#!/usr/bin/env python3
"""Runtime adaptive tutor for curriculum_graph_builder.py JSON graphs.

Set OPENROUTER_API_KEY in .env or the environment; --api-key can override it.
"""
import argparse
import json
import os
import sys
import textwrap
import requests
import random
import re
from datetime import datetime, timezone
from collections import deque

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _load_dotenv(path=".env"):
    """Load simple .env assignments without overriding existing environment values."""
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if value[:1] in ("\"", "'") and value[-1:] == value[:1]:
                    value = value[1:-1]
                elif " #" in value:
                    value = value.split(" #", 1)[0].rstrip()
                if key:
                    os.environ.setdefault(key, value)
    except OSError:
        # A .env file is optional; environment variables still work.
        pass


_load_dotenv()


def log(message):
    """Print an ASCII-safe status line."""
    safe = str(message).encode("ascii", "backslashreplace").decode("ascii")
    print("[tutor] " + safe)


def progress(done, total, width=24):
    total = max(1, total)
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "] %d/%d" % (done, total)


def load_graph(path):
    with open(path, "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("Graph JSON must contain a nodes list.")
    nodes = graph["nodes"]
    ids = set()
    for node in nodes:
        if not isinstance(node, dict) or not node.get("id"):
            raise ValueError("Every graph node must be an object with an id.")
        node["id"] = str(node["id"])
        if node["id"] in ids:
            raise ValueError("Duplicate node id: " + node["id"])
        ids.add(node["id"])
        node["prerequisites"] = node.get("prerequisites") or []
        if not isinstance(node["prerequisites"], list):
            raise ValueError("Prerequisites must be a list for " + node["id"])
    for node in nodes:
        missing = [p for p in node["prerequisites"] if p not in ids]
        if missing:
            raise ValueError("Unknown prerequisite(s) for %s: %s" % (node["id"], ", ".join(missing)))
    # Reject cycles; even a fully mastered cycle should not silently deadlock.
    pending = {n["id"]: set(n["prerequisites"]) for n in nodes}
    ready = deque(key for key, reqs in pending.items() if not reqs)
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for key, reqs in pending.items():
            if current in reqs:
                reqs.remove(current)
                if not reqs:
                    ready.append(key)
    if visited != len(nodes):
        raise ValueError("Graph contains a prerequisite cycle.")
    return graph, nodes


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(text[start:end + 1])


def _topic_text(node):
    return (str(node.get("title", "")) + " " + str(node.get("description", ""))).casefold()

def _topic_assessment_history(profile, topic_id):
    events = profile.get("history", []) if isinstance(profile, dict) else []
    return [e for e in events if isinstance(e, dict) and e.get("topic_id") == topic_id
            and e.get("event") in ("pre_test", "quiz_attempt") and e.get("score") in (0, 1)]

def choose_question_type(node, profile=None):
    """Choose a question format from topic intent and recent learner evidence.

    New/foundational concepts get a cloze scaffold, applied topics get a
    scenario, errors temporarily restore a scaffold, and a two-answer correct
    streak earns an unaided short-answer question.
    """
    text = _topic_text(node)
    history = _topic_assessment_history(profile or {}, node.get("id"))
    if history:
        # A recent miss calls for an immediately supported check. A retry that
        # succeeds naturally returns to applied practice; sustained success
        # moves the learner to unaided recall.
        if history[-1].get("score") == 0:
            return "cloze_mcq"
        streak = 0
        for event in reversed(history):
            if event.get("score") != 1:
                break
            streak += 1
        if streak >= 2:
            return "free_text"

    # Definitions and introductory/foundation topics are well suited to a
    # completion prompt. Practice-oriented descriptions should remain scenarios.
    concept = bool(re.search(
        r"\b(introduction|introductory|foundation|define|definition|meaning|"
        r"terminology|vocabulary|key concept|what is)\b", text))
    applied = bool(re.search(
        r"\b(use|create|generate|write|apply|application|prompt|tool|debug|"
        r"troubleshoot|workflow|implement|configure|build|design|produce)\b", text))
    if concept and not applied:
        return "cloze_mcq"
    if concept:
        # Introductory topics still start with a low-friction concept check.
        return "cloze_mcq"
    if applied:
        return "scenario_mcq"
    # Deterministic fallback for sparse metadata; never collapse the whole
    # curriculum to one question format merely because keywords are absent.
    return "cloze_mcq"

def _normalize_answer(value):
    import unicodedata
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())

def grade_answer(question, response):
    """Grade an option index or a normalized short-text answer."""
    if question.get("type") == "free_text":
        candidate = _normalize_answer(response)
        accepted = {_normalize_answer(a) for a in question.get("accepted_answers", [])}
        return bool(candidate) and candidate in accepted
    try:
        return int(response) == question["answer"]
    except (ValueError, TypeError, KeyError):
        return False


class ContentProvider:
    def __init__(self, api_key=None, model=None, simulate=False):
        # CLI values win; otherwise use environment/.env, then the built-in model.
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
        self.simulate = bool(simulate)
        if not self.simulate and not self.api_key:
            raise RuntimeError("OpenRouter API key not found. Set OPENROUTER_API_KEY in .env or the environment, or use --simulate.")
        if self.simulate:
            log("OFFLINE MODE: --simulate was selected.")
        else:
            log("LIVE MODE: dynamic content is enabled via OpenRouter model %s." % self.model)

    def ask(self, prompt, system):
        """Make a live OpenRouter request; never silently substitute mock content."""
        if self.simulate or not self.api_key:
            raise RuntimeError("Internal error: OpenRouter called without live mode and API key.")
        payload = {"model": self.model, "temperature": 0.75,
                   "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": prompt}]}
        headers = {"Authorization": "Bearer " + self.api_key,
                   "Content-Type": "application/json",
                   "HTTP-Referer": "https://openrouter.ai/",
                   "X-Title": "Runtime Adaptive Tutor"}
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter request failed (no offline fallback in live mode): %s" % exc) from exc

    @staticmethod
    def mock_question(node, kind):
        title = node.get("title", node["id"].replace("_", " ").title())
        desc = node.get("description", "Apply the main idea of this topic.")
        qtype = kind if kind in ("scenario_mcq", "cloze_mcq", "free_text") else "scenario_mcq"
        if qtype == "free_text":
            return {"type": qtype, "question": "In one or two words, name the key concept for %s." % title,
                    "accepted_answers": [title, node["id"].replace("_", " ")],
                    "explanation": "The key concept is %s: %s" % (title, desc)}
        if qtype == "cloze_mcq":
            prompt = "Complete the statement about %s: ____" % title
        else:
            prompt = ("Scenario: You must achieve this practical goal: %s. "
                      "Given the relevant skills (%s), what is the best next action?" % (desc, title))
        return {"type": qtype, "question": prompt,
                "options": ["Apply the idea to the task: " + desc, "Ignore the task requirements and choose at random.",
                            "Assume the tool is always correct without checking.", "Use an unrelated technique that does not address the goal."],
                "answer": 0, "explanation": "This applies the topic directly: " + desc,
                "distractor_feedback": ["Correct: this choice uses the topic to meet the stated goal.",
                    "Random choices do not use evidence or the topic's method.", "Outputs should be checked; tools can make mistakes.",
                    "An unrelated technique does not solve the stated task."]}

    def question(self, node, kind, profile=None):
        # Callers may pass an explicit type or a purpose label (e.g. "baseline
        # diagnostic"); purpose labels must not silently force scenario_mcq.
        qtype = kind if kind in ("scenario_mcq", "cloze_mcq", "free_text") else choose_question_type(node, profile)
        if self.simulate:
            return validate_question(self.mock_question(node, qtype))
        log("[LLM] Generating %s question for '%s' via OpenRouter..." % (qtype, node["id"]))
        common = ('Topic: %s. Description: %s. Purpose: %s. Explicit target type: %s. ' %
                  (node.get("title", node["id"]), node.get("description", ""), kind, qtype))
        if qtype == "free_text":
            spec = ('Ask for one unambiguous term/short answer only (not an essay). Return strict JSON with keys '
                    '"type":"free_text", "question", "accepted_answers" (nonempty array of exact equivalent aliases), "explanation". '
                    'No options and no answer index. Keep aliases precise; do not include broad synonyms.')
        elif qtype == "cloze_mcq":
            spec = ('Return strict JSON with keys "type":"cloze_mcq", "question" containing exactly one ____ blank, '
                    '"options" (exactly 4 strings), "answer" (zero-based integer 0-3), "explanation", '
                    'and "distractor_feedback" (4 strings).')
        else:
            spec = ('Write a concrete applied scenario, not a generic definition. Return strict JSON with keys '
                    '"type":"scenario_mcq", "question", "options" (exactly 4 strings), "answer" (zero-based integer 0-3), '
                    '"explanation", and "distractor_feedback" (4 strings).')
        result = _extract_json(self.ask(common + spec, "You are a precise instructional designer. Follow target type exactly; return JSON only."))
        return validate_question(result, expected_type=qtype)

    def lesson(self, node, learner_context="", fast_track=False):
        if self.simulate:
            depth = "Fast track (prior mastery): concise recap, one nuance, and a quick application." if fast_track else "Scaffolded lesson: build the concept step by step, then apply it."
            return {"lesson": "%s: %s %s Try applying this idea to a concrete task, then check whether the result meets the goal." %
                    (node.get("title", node["id"]), node.get("description", ""), depth),
                    "tip": "Tip: state the goal clearly, try one small example, and verify the result."}
        log("[LLM] Generating real-time micro-lesson for '%s' via OpenRouter..." % node["id"])
        depth = ("Use a concise fast-track recap (roughly 50-80 words), clarify one nuance, and still teach/test this topic; do not omit it. "
                 if fast_track else "Use a scaffolded micro-lesson (roughly 100-150 words) that explains the concept and one example. ")
        prompt = ("Write a crisp, tailored, interactive micro-lesson for topic %s: %s. %s"
                  "Include one question or action for the learner and a practical concept tip. "
                  "Adapt to this learner context: %s. Return JSON only with keys lesson and tip." %
                  (node.get("title", node["id"]), node.get("description", ""), depth, learner_context))
        result = _extract_json(self.ask(prompt, "You are an expert tutor. Keep language clear and supportive."))
        lesson = str(result.get("lesson", ""))
        if len(lesson.split()) > 165:
            lesson = " ".join(lesson.split()[:150])
        return {"lesson": lesson, "tip": str(result.get("tip", ""))}

    def remediation(self, node, question, selected, feedback):
        if self.simulate:
            return "Review the goal and compare each option with the topic description. Identify which choice applies the idea directly, then try again."
        log("[LLM] Generating real-time remediation hint and explanation for '%s' via OpenRouter..." % node["id"])
        prompt = ("The learner missed this question about %s. Question: %s. Their answer: %s. "
                  "Feedback: %s. Give one concise, non-revealing hint and then a deeper explanation "
                  "that helps them retry. Return JSON only with keys hint and explanation." %
                  (node.get("title", node["id"]), question, selected, feedback))
        result = _extract_json(self.ask(prompt, "You are a supportive tutor. Do not shame the learner."))
        return str(result.get("hint", "Try comparing each option to the learning goal.")) + "\n" + str(result.get("explanation", ""))


def validate_question(question, expected_type=None):
    if not isinstance(question, dict):
        raise ValueError("Question response must be a JSON object.")
    qtype = question.get("type")
    if qtype not in ("scenario_mcq", "cloze_mcq", "free_text") or (expected_type and qtype != expected_type):
        raise ValueError("Question type is missing or does not match the requested type.")
    if not isinstance(question.get("question"), str) or not question["question"].strip():
        raise ValueError("Question text must be nonempty.")
    if qtype == "free_text":
        answers = question.get("accepted_answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError("Free-text questions need nonempty accepted_answers.")
        if "options" in question or "answer" in question:
            raise ValueError("Free-text questions must not contain options or an answer index.")
        question["accepted_answers"] = list(dict.fromkeys(a.strip() for a in answers))
    else:
        options, answer = question.get("options"), question.get("answer")
        if not isinstance(options, list) or len(options) != 4 or any(not isinstance(x, str) or not x.strip() for x in options):
            raise ValueError("Choice questions must have exactly four nonempty options.")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
            raise ValueError("Question answer must be an option index from 0 to 3.")
        if qtype == "cloze_mcq" and question["question"].count("____") != 1:
            raise ValueError("Cloze question must contain exactly one ____ blank.")
        feedback = question.get("distractor_feedback", [""] * 4)
        if not isinstance(feedback, list) or len(feedback) != 4:
            feedback = ["Review how this option relates to the learning goal." for _ in range(4)]
        question["distractor_feedback"] = feedback
    question["explanation"] = str(question.get("explanation", "Review the topic and try again."))
    return question

def present_question(question):
    print("\n" + question["question"])
    for index, option in enumerate(question.get("options", []), 1):
        print("  %d) %s" % (index, option))

def get_choice(question, simulated=False, simulated_correct=True):
    present_question(question)
    if simulated:
        if question["type"] == "free_text":
            response = question["accepted_answers"][0] if simulated_correct else "unmatched response"
            log("Simulated textual answer.")
            return response
        choice = question["answer"] if simulated_correct else (question["answer"] + 1) % 4
        log("Simulated answer: %d" % (choice + 1))
        return choice
    if question["type"] == "free_text":
        return input("Your short answer: ").strip()
    while True:
        raw = input("Your answer (1-4): ").strip()
        if raw in ("1", "2", "3", "4"):
            return int(raw) - 1
        print("Please enter a number from 1 to 4.")



# Transparent FSRS-style memory scheduling (stability is measured in days).
def _as_utc_datetime(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError("now must be a datetime, ISO timestamp, or None")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

def normalize_fsrs_rating(rating):
    """Normalize FSRS ratings 1..4 or correctness scores 0..1 to 1..4.

    Conventions (FSRS/SM-2): Again=1, Hard=2, Good=3, Easy=4; scores map
    0->Again, <0.6->Hard, <0.85->Good, >=0.85->Easy. Ambiguous inputs are
    resolved deliberately: bool True -> Good(3) / False -> Again(1) (booleans
    are outcomes, not ratings -- float(True)==1.0 would otherwise read as a
    perfect score -> Easy); int 1 -> Again(1) (checked before the score range,
    since an explicit rating beats the coincidental overlap with score 1.0);
    float 1.0 -> Easy(4) (a perfect numeric score).
    """
    if isinstance(rating, bool):
        return 3 if rating else 1
    # Exact integer ratings take priority over the overlapping 0..1 score band.
    if isinstance(rating, int) and 1 <= rating <= 4:
        return rating
    value = float(rating)
    if 0.0 <= value <= 1.0:
        # Correctness score maps to Again / Hard / Good / Easy.
        return 1 if value < .4 else 2 if value < .6 else 3 if value < .85 else 4
    if value.is_integer() and 1 <= int(value) <= 4:
        return int(value)
    raise ValueError("rating must be an FSRS rating (1..4) or score (0..1)")

def calculate_difficulty(previous_difficulty, rating):
    rating = normalize_fsrs_rating(rating)
    previous = 5.0 if previous_difficulty is None else float(previous_difficulty)
    return round(max(1.0, min(10.0, previous + (3 - rating) * .55)), 3)

def calculate_stability(previous_stability, difficulty, rating):
    rating = normalize_fsrs_rating(rating)
    previous = max(.1, float(previous_stability if previous_stability is not None else .5))
    difficulty = max(1.0, min(10.0, float(difficulty)))
    if rating == 1:
        return round(max(.1, previous * .35), 3)
    growth = {2: 1.25, 3: 2.25, 4: 3.2}[rating]
    return round(max(.5, previous * growth * (1.15 - (difficulty - 5) * .035)), 3)

def calculate_retrievability(stability, elapsed_days):
    """Estimate recall probability using the FSRS power forgetting curve."""
    stability = max(.1, float(stability))
    elapsed_days = max(0.0, float(elapsed_days))
    return round((1.0 + elapsed_days / (9.0 * stability)) ** -1.0, 6)

def calculate_interval(stability, rating=3):
    rating = normalize_fsrs_rating(rating)
    stability = max(.1, float(stability))
    return max(1, int(round(stability * ({1: .25, 2: .8, 3: 1.0, 4: 1.25}[rating]))))

def calculate_next_due_timestamp(now, interval_days):
    return (_as_utc_datetime(now) + __import__("datetime").timedelta(days=max(1, int(interval_days)))).isoformat()

def fsrs_review_update(review, rating, now=None, score=None):
    """Return updated per-topic review state; accepts rating 1..4 or score 0..1."""
    now = _as_utc_datetime(now)
    review = dict(review or {})
    normalized = normalize_fsrs_rating(rating)
    old_stability = max(.1, float(review.get("stability", .5)))
    difficulty = calculate_difficulty(review.get("difficulty", 5.0), normalized)
    stability = calculate_stability(old_stability, difficulty, normalized)
    interval = calculate_interval(stability, normalized)
    previous_due = review.get("due_date")
    elapsed = 0.0
    if review.get("last_reviewed"):
        elapsed = max(0.0, (now - _as_utc_datetime(review["last_reviewed"])).total_seconds() / 86400.0)
    state = {"stability": stability, "difficulty": difficulty,
             "due_date": calculate_next_due_timestamp(now, interval),
             "last_reviewed": now.isoformat(), "reps": int(review.get("reps", 0)) + 1,
             "last_rating": normalized, "interval_days": interval,
             "retrievability": calculate_retrievability(old_stability, elapsed)}
    if score is not None:
        state["last_score"] = max(0.0, min(1.0, float(score)))
    return state

class SpacedRepetitionEngine:
    """Profile-backed per-topic review scheduler shared by CLI and web clients."""
    def __init__(self, profile, profile_path=None, nodes=None):
        self.profile = profile
        self.profile_path = profile_path
        self.node_by_id = {str(n["id"]): n for n in (nodes or []) if isinstance(n, dict) and n.get("id")}
        reviews = profile.setdefault("fsrs_reviews", {})
        if not isinstance(reviews, dict):
            reviews = profile["fsrs_reviews"] = {}
        # Import legacy web state without discarding either structure.
        for topic_id, value in (profile.get("topic_reviews") or {}).items():
            reviews.setdefault(topic_id, value)
        profile["topic_reviews"] = reviews  # backward-compatible alias in JSON

    def get_due_reviews(self, now=None):
        current = _as_utc_datetime(now)
        output = []
        for topic_id, state in self.profile.get("fsrs_reviews", {}).items():
            if not isinstance(state, dict) or not state.get("due_date"):
                continue
            try:
                due = _as_utc_datetime(state["due_date"] )
            except (ValueError, TypeError, OverflowError):
                continue
            if due <= current:
                node = self.node_by_id.get(topic_id, {})
                lag = max(0.0, (current - due).total_seconds() / 86400.0)
                last = state.get("last_reviewed")
                elapsed = max(0.0, (current - _as_utc_datetime(last)).total_seconds() / 86400.0) if last else 0.0
                output.append({"topic_id": topic_id, "title": node.get("title", topic_id),
                    "due_date": due.isoformat(), "stability": state.get("stability"),
                    "difficulty": state.get("difficulty"), "reps": state.get("reps", 0),
                    "retrievability": calculate_retrievability(state.get("stability", .5), elapsed),
                    "explanation": "Review is due: scheduled timestamp has passed by %.2f day(s)." % lag,
                    "xai": {"reason": "due_date_passed", "days_overdue": round(lag, 2),
                            "retrievability": calculate_retrievability(state.get("stability", .5), elapsed)}})
        return sorted(output, key=lambda item: item["due_date"])

    def record_review(self, topic_id, rating=None, score=None, now=None, details=None):
        topic_id = str(topic_id)
        if rating is None and score is None:
            raise ValueError("Provide rating (1..4) or score (0..1)")
        raw = score if rating is None else rating
        normalized = normalize_fsrs_rating(raw)
        state = fsrs_review_update(self.profile.get("fsrs_reviews", {}).get(topic_id), normalized, now, score)
        self.profile["fsrs_reviews"][topic_id] = state
        self.profile["topic_reviews"] = self.profile["fsrs_reviews"]
        attempt = {"timestamp": state["last_reviewed"], "event": "spaced_repetition_review",
                   "topic_id": topic_id, "score": score, "rating": normalized,
                   "interval_days": state["interval_days"], "due_date": state["due_date"],
                   "stability": state["stability"], "difficulty": state["difficulty"],
                   "xai": {"reason": "review_attempt_recorded", "rating": normalized,
                           "interval_days": state["interval_days"]}}
        if details:
            attempt["details"] = details
        self.profile.setdefault("history", []).append(attempt)
        if self.profile_path:
            save_profile(self.profile_path, self.profile)
        return dict(state)

def explain_topic_scheduling(node, profile, due_reviews=None):
    """Machine-readable explanation for selecting a topic in the curriculum."""
    topic_id = str(node.get("id", ""))
    mastered = set(profile.get("mastered_topics", []))
    missing = [dep for dep in node.get("prerequisites", []) if dep not in mastered]
    due = next((item for item in (due_reviews or []) if item.get("topic_id") == topic_id), None)
    if due:
        return {"code": "due_review", "reason": "spaced_repetition_due",
                "explanation": due["explanation"], "evidence": due.get("xai", {})}
    if missing:
        return {"code": "prerequisites_locked", "reason": "prerequisites_incomplete",
                "explanation": "Topic is not schedulable until its prerequisites are mastered.", "evidence": {"missing_prerequisites": missing}}
    return {"code": "curriculum_order", "reason": "first_eligible_topic",
            "explanation": "Topic is the first unmastered topic whose prerequisites are complete.",
            "evidence": {"prerequisites": list(node.get("prerequisites", [])), "mastered_count": len(mastered)}}


def eligible_nodes(nodes, mastered):
    return [node for node in nodes
            if node["id"] not in mastered and all(p in mastered for p in node["prerequisites"])]


def profile_path(args):
    if args.profile_file:
        return os.path.abspath(args.profile_file)
    if args.user_id:
        safe_id = "".join(ch for ch in args.user_id if ch.isalnum() or ch in "-_ .").strip().replace(" ", "_")
        if not safe_id:
            raise ValueError("--user-id must contain at least one safe filename character.")
        return os.path.abspath("learner_profile_%s.json" % safe_id)
    return os.path.abspath("learner_profile.json")


def load_profile(path, user_id=None):
    if not os.path.exists(path):
        return {"user_id": user_id, "mastered_topics": [], "history": [], "misconceptions": {}, "fsrs_reviews": {}, "updated_at": None}
    with open(path, "r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise ValueError("Learner profile must be a JSON object.")
    profile.setdefault("user_id", user_id)
    profile.setdefault("mastered_topics", [])
    profile.setdefault("history", [])
    profile.setdefault("misconceptions", {})
    profile.setdefault("portfolio", {})
    profile.setdefault("chapter_quizzes", {})
    profile.setdefault("fsrs_reviews", profile.get("topic_reviews", {}))
    if not isinstance(profile["fsrs_reviews"], dict):
        profile["fsrs_reviews"] = {}
    profile["topic_reviews"] = profile["fsrs_reviews"]
    # Preserve all legacy keys while filling in newly introduced structures.
    if not isinstance(profile["portfolio"], dict):
        profile["portfolio"] = {}
    if not isinstance(profile["chapter_quizzes"], dict):
        profile["chapter_quizzes"] = {}
    if not isinstance(profile["mastered_topics"], list) or not isinstance(profile["history"], list):
        raise ValueError("Profile mastered_topics and history must be lists.")
    return profile


def save_profile(path, profile):
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def record_event(profile, path, event, topic, score=None, misconception=None, details=None):
    timestamp = datetime.now(timezone.utc).isoformat()
    item = {"timestamp": timestamp, "event": event, "topic_id": topic, "score": score,
            "misconceptions": [misconception] if misconception else []}
    if details:
        item.update(details)
    profile["history"].append(item)
    if misconception:
        by_topic = profile.setdefault("misconceptions", {})
        by_topic.setdefault(topic, []).append({"timestamp": timestamp, "text": misconception})
    profile["mastered_topics"] = sorted(set(profile.get("mastered_topics", [])))
    save_profile(path, profile)


def _chapter_for(node):
    for key in ("chapter", "chapter_id", "module", "module_id", "unit"):
        value = node.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "Course"


def _chapter_groups(nodes):
    groups = {}
    for node in nodes:
        key = _chapter_for(node)
        groups.setdefault(key, []).append(node)
    return groups


def collect_portfolio(profile, path, force=False, simulated=False):
    """Collect and persist learner identity, goals, context, and preferences."""
    fields = (("name", "نام یا نحوهٔ خطاب"), ("goal", "هدف اصلی یادگیری"),
              ("background", "پیش‌زمینه یا سطح فعلی"), ("preferences", "ترجیحات یادگیری / زبان"))
    portfolio = profile.setdefault("portfolio", {})
    if not isinstance(portfolio, dict):
        portfolio = profile["portfolio"] = {}
    missing = not all(str(portfolio.get(key, "")).strip() for key, _ in fields)
    if not force and not missing:
        return portfolio
    if simulated:
        defaults = {"name": profile.get("user_id") or "Learner", "goal": "Complete the curriculum and apply it in practice",
                    "background": "To be determined from diagnostic", "preferences": "Adaptive micro-lessons and varied quizzes"}
        for key, _label in fields:
            if force or not str(portfolio.get(key, "")).strip():
                portfolio[key] = defaults[key]
    else:
        print("\nLearner portfolio — helps personalize the course.")
        for key, label in fields:
            current = str(portfolio.get(key, ""))
            if force or not current.strip():
                answer = input("%s%s: " % (label, (" [%s]" % current) if current else "")).strip()
                if answer:
                    portfolio[key] = answer
    profile["portfolio"] = portfolio
    save_profile(path, profile)
    return portfolio


def _diagnostic_batches(nodes, requested_count):
    """Build a small set of multi-topic scenarios balanced across course modules."""
    if not nodes:
        return []
    try:
        count = int(requested_count)
    except (TypeError, ValueError):
        count = 4
    count = min(12, max(1, count), len(nodes))
    # Interleave modules before sampling, so early graph ordering does not
    # monopolize the diagnostic. Each scenario covers at most two topics.
    modules = {}
    for node in nodes:
        modules.setdefault(_chapter_for(node), []).append(node)
    ordered = []
    while any(modules.values()):
        for module_nodes in modules.values():
            if module_nodes:
                ordered.append(module_nodes.pop(0))
    covered_count = min(len(nodes), count * 2)
    if len(nodes) > count:
        covered_count = min(covered_count, len(nodes) - 1)
    selected = ordered[:covered_count]
    batches = [[] for _ in range(min(count, len(selected)))]
    for index, node in enumerate(selected):
        batches[index % len(batches)].append(node)
    return [batch for batch in batches if batch]


def _diagnostic_node(batch, index):
    titles = [str(node.get("title", node["id"])) for node in batch]
    details = ["%s: %s" % (node.get("title", node["id"]), node.get("description", "apply the core skill")) for node in batch]
    return {"id": "diagnostic_scenario_%d" % (index + 1),
            "title": "Scenario applying " + ", ".join(titles),
            "description": ("Create one realistic, concrete application scenario that requires the learner to use these relevant skills together: "
                            + "; ".join(details) + ". Ask which action best solves the situation, not for definitions.")}


def _mastery_level(profile, topic_id):
    attempts = _topic_assessment_history(profile, topic_id)
    if attempts and attempts[-1].get("score") == 1:
        return "fast"
    # Include correctly passed diagnostic as formative evidence; never mark mastery here.
    pre = [e for e in profile.get("history", []) if isinstance(e, dict) and e.get("topic_id") == topic_id and e.get("event") == "pre_test"]
    return "fast" if pre and pre[-1].get("score") == 1 else "full"


def _question_feedback(question, response, graded):
    if graded is None:
        return "پاسخ متنی را با اطمینان نمی‌توانم نمره‌گذاری کنم؛ لطفاً خودارزیابی کنید یا پاسخ را روشن‌تر بازنویسی کنید."
    if graded:
        return question.get("explanation", "پاسخ درست است.")
    if isinstance(response, int):
        return question.get("distractor_feedback", [])[response] if response < len(question.get("distractor_feedback", [])) else question.get("explanation", "پاسخ را با هدف سؤال مقایسه کنید.")
    return question.get("explanation", "پاسخ با کلیدواژه‌های مورد انتظار تطبیق نداشت؛ ممکن است بیان دیگری درست باشد.")


def _record_srs_review(scheduler, topic_id, graded, score=None, details=None):
    """Feed a graded attempt into the FSRS scheduler; never let it break the lesson flow.

    Explicit 1..4 integer ratings are passed through untouched. Boolean/None
    outcomes (correct / incorrect / uncertain) map directly to FSRS ratings:
    True -> Good(3), False -> Again(1), None -> Hard(2) so uncertain answers
    reschedule sooner than normal recalls. Uncertain reviews also get an XAI
    hint in their stored attempt. The numeric score channel is reserved for
    real scores: passing booleans there silently coerces via float() (True ->
    Easy, False -> Again), so graded outcomes always go through `rating`.
    """
    if scheduler is None or not topic_id:
        return None
    try:
        details = dict(details) if details else {}
        if isinstance(graded, int) and not isinstance(graded, bool) and 1 <= graded <= 4:
            rating = graded
        elif graded is True:
            rating = 3
        elif graded is False:
            rating = 1
        else:
            rating = 2
            details.setdefault("xai_hint", "uncertain_answer_rescheduled_with_hard_rating")
        # Pass score only when it is a real numeric score; never a boolean outcome.
        state = scheduler.record_review(topic_id, rating=rating,
                                        score=(float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None),
                                        details=details)
        log("FSRS update for %s: rating=%d, interval=%dd, next review %s" %
            (topic_id, state["last_rating"], state["interval_days"], state["due_date"]))
        return state
    except (ValueError, TypeError, OSError) as exc:
        log("FSRS review could not be recorded for %s: %s" % (topic_id, exc))
        return None


def _run_due_reviews(provider, profile, path, nodes_by_id, scheduler, due_reviews):
    """Serve due spaced-repetition reviews before new material.

    Each due topic gets one short recall question; the outcome updates the
    FSRS state via record_review (a lapse triggers a remediation micro-lesson).
    """
    if not due_reviews:
        return
    print("\nSpaced repetition: %d review(s) are due before new lessons." % len(due_reviews))
    for item in due_reviews:
        topic_id = str(item.get("topic_id", ""))
        node = nodes_by_id.get(topic_id)
        title = item.get("title", topic_id)
        print("\nDue review: %s | %s" % (title, item.get("explanation", "")))
        if not node:
            log("Skipping due review; topic no longer exists in the graph: " + topic_id)
            profile.get("fsrs_reviews", {}).pop(topic_id, None)
            save_profile(path, profile)
            continue
        try:
            question = provider.question(node, "cloze_mcq", profile)
        except (ValueError, KeyError, TypeError) as exc:
            log("Review question unavailable for %s (%s); rescheduling tomorrow." % (topic_id, exc))
            _record_srs_review(scheduler, topic_id, None, details={"source": "spaced_repetition_review",
                                                                   "reason": "question_generation_failed"})
            continue
        response = get_choice(question, provider.simulate, True)
        graded = grade_answer(question, response)
        feedback = _question_feedback(question, response, graded)
        print(("Recalled. " if graded else "Review. ") + str(feedback))
        if graded is False:
            try:
                lesson = provider.lesson(node, "Spaced-repetition review showed forgetting; rebuild the core idea.", fast_track=True)
                print("Refresher micro-lesson: " + lesson["lesson"])
            except (ValueError, KeyError, TypeError) as exc:
                log("Refresher lesson unavailable: %s" % exc)
        record_event(profile, path, "review_quiz_attempt", topic_id,
                     1 if graded is True else (0 if graded is False else None),
                     None if graded else feedback,
                     {"source": "spaced_repetition_review", "question": question["question"],
                      "selected_option": response if isinstance(response, int) else None,
                      "response": response if isinstance(response, str) else None})
        _record_srs_review(scheduler, topic_id, graded, details={"source": "spaced_repetition_review",
                                                                 "question_type": question["type"]})


def _chapter_quiz(provider, profile, path, chapter, chapter_nodes, question_count, scheduler=None):
    states = profile.setdefault("chapter_quizzes", {})
    state = states.setdefault(chapter, {"status": "pending", "attempts": [], "completed": False})
    if state.get("completed"):
        log("Chapter quiz already completed; not repeating: " + chapter)
        return
    if state.get("status") != "in_progress":
        state.update({"status": "in_progress", "started_at": datetime.now(timezone.utc).isoformat()})
        state.setdefault("attempts", [])
        save_profile(path, profile)
    count = max(1, int(question_count))
    titles = [n.get("title", n["id"]) for n in chapter_nodes]
    log("Chapter challenge quiz: %s | %d questions" % (chapter, count))
    # Preserve question-by-question progress so a stopped session can resume safely.
    answers = state.setdefault("current_answers", [])
    correct_count = sum(1 for item in answers if item.get("correct") is True)
    uncertain = sum(1 for item in answers if item.get("correct") is None)
    for index in range(len(answers), count):
        node = chapter_nodes[index % len(chapter_nodes)]
        kinds = ("scenario_mcq", "cloze_mcq", "free_text")
        kind = kinds[index % len(kinds)]
        question = provider.question(node, kind, profile)
        print("\nChapter %s — question %d/%d (topic: %s)" % (chapter, index + 1, count, node.get("title", node["id"])))
        response = get_choice(question, provider.simulate, True)
        graded = grade_answer(question, response)
        if graded is None and not provider.simulate:
            print("Feedback: " + _question_feedback(question, response, graded))
            self_rating = input("Your self-rating: [1] confident/correct, [2] unsure/incorrect, [R] retry: ").strip().lower()
            if self_rating == "r":
                response = get_choice(question, False)
                graded = grade_answer(question, response)
                if graded is None:
                    self_rating = input("Still uncertain. Self-rate [1] correct / [2] incorrect: ").strip()
            if graded is None:
                graded = True if self_rating == "1" else False
        correct_count += int(graded is True)
        uncertain += int(graded is None)
        feedback = _question_feedback(question, response, graded)
        print(("Correct. " if graded else "Review. ") + str(feedback))
        item = {"question": question["question"], "question_type": question["type"], "topic_id": node["id"],
                "response": response if isinstance(response, str) else None,
                "selected_option": response if isinstance(response, int) else None,
                "correct": graded, "feedback": feedback}
        answers.append(item)
        state["attempts"].append(item)
        record_event(profile, path, "chapter_quiz_attempt", node["id"], 1 if graded is True else (0 if graded is False else None),
                     None if graded else feedback, {"chapter": chapter, "question_type": question["type"], "question": question["question"]})
        _record_srs_review(scheduler, node["id"], graded, details={"source": "chapter_quiz", "chapter": chapter,
                                                                   "question_type": question["type"]})
    score = correct_count / float(count)
    threshold = float(getattr(provider, "chapter_pass_threshold", 0.7))
    weak = score < threshold
    state.update({"status": "completed", "completed": True, "score": score, "correct": correct_count,
                  "question_count": count, "weak": weak, "completed_at": datetime.now(timezone.utc).isoformat(),
                  "uncertain_count": uncertain})
    save_profile(path, profile)
    log("Chapter result: %d/%d (%.0f%%)%s" % (correct_count, count, score * 100, " — revisit recommended" if weak else ""))
    if weak:
        weak_topics = []
        for item in answers[-count:]:
            if item.get("correct") is not True and item.get("topic_id") not in weak_topics:
                weak_topics.append(item.get("topic_id"))
        by_id = {n["id"]: n for n in chapter_nodes}
        state["remediation_topics"] = weak_topics
        for topic_id in weak_topics:
            node = by_id.get(topic_id)
            if not node:
                continue
            print("\nRevisiting lesson: " + node.get("title", topic_id))
            lesson = provider.lesson(node, "Chapter quiz showed an area to strengthen. Learner portfolio: %s" % profile.get("portfolio", {}), fast_track=False)
            print("Remediation lesson: " + lesson["lesson"])
            print("Concept tip: " + lesson["tip"])
            record_event(profile, path, "chapter_remediation", topic_id, None, details={"chapter": chapter, "lesson": lesson["lesson"]})
        state["remediation_completed"] = True
        save_profile(path, profile)


def run(args):
    graph, nodes = load_graph(args.graph)
    provider = ContentProvider(args.api_key, args.model, args.simulate)
    course = graph.get("course_name", "Curriculum")
    pfile = profile_path(args)
    profile = load_profile(pfile, args.user_id)
    scheduler = SpacedRepetitionEngine(profile, pfile, nodes)
    due = scheduler.get_due_reviews()
    if getattr(args, "check_due_reviews", False):
        print(json.dumps(due, ensure_ascii=False, indent=2))
        return 0
    log("Due spaced-repetition reviews: %d" % len(due))
    collect_portfolio(profile, pfile, force=bool(getattr(args, "update_portfolio", False)), simulated=provider.simulate)
    valid_ids = {n["id"] for n in nodes}
    mastered = set(profile.get("mastered_topics", [])) & valid_ids
    profile["mastered_topics"] = sorted(mastered)
    returning = bool(profile.get("history") or mastered)
    log("Course: %s | Topics: %d" % (course, len(nodes)))
    if returning:
        log("Welcome back%s! Resuming: %d/%d topics completed." % ((" " + str(profile.get("user_id"))) if profile.get("user_id") else "", len(mastered), len(nodes)))
        log("Loaded learner profile: " + pfile)
    else:
        log("Starting a new learner journey. Profile: " + pfile)

    # Serve overdue FSRS reviews first so memory strengthening precedes new material.
    _run_due_reviews(provider, profile, pfile, {n["id"]: n for n in nodes}, scheduler, due)

    # A compact scenario diagnostic informs lesson depth only; it never omits curriculum nodes.
    diag = profile.get("diagnostic_state")
    if not isinstance(diag, dict):
        diag = {}
        profile["diagnostic_state"] = diag
    legacy_pretests = [e for e in profile.get("history", []) if isinstance(e, dict) and e.get("event") == "pre_test"]
    legacy_unindexed = [e for e in legacy_pretests if "diagnostic_question_index" not in e]
    # Older versions assessed every node individually. Preserve those results without
    # confusing an interrupted modern scenario diagnostic for a completed old one.
    if legacy_unindexed and "completed" not in diag:
        diag.update({"completed": True, "legacy_imported": True, "attempts": len(legacy_unindexed)})
        save_profile(pfile, profile)
    if mastered and not diag.get("completed"):
        # Returning learners continue their saved course; unseen topics get ordinary lessons.
        diag.update({"completed": True, "skipped_for_resume": True, "attempts": 0})
        save_profile(pfile, profile)
    diagnostic_count = getattr(args, "diagnostic_question_count", 4)
    batches = _diagnostic_batches(nodes, diagnostic_count)
    completed_indices = set(diag.get("completed_question_indices", []))
    if not diag.get("completed") and batches:
        log("Starting scenario diagnostic: %d questions for %d topics; all topics remain in the course." % (len(batches), len(nodes)))
        for index, batch in enumerate(batches):
            if index in completed_indices:
                continue
            scenario_node = _diagnostic_node(batch, index)
            print("\nScenario diagnostic %s (covers: %s)" %
                  (progress(index + 1, len(batches)), ", ".join(n.get("title", n["id"]) for n in batch)))
            try:
                question = provider.question(scenario_node, "scenario_mcq", profile)
            except (ValueError, KeyError, TypeError) as exc:
                log("Diagnostic scenario invalid; using local scenario fallback: %s" % exc)
                question = validate_question(provider.mock_question(scenario_node, "scenario_mcq"), expected_type="scenario_mcq")
            simulated_correct = (index % 2 == 0)
            response = get_choice(question, provider.simulate, simulated_correct)
            graded = grade_answer(question, response)
            misconception = None if graded is not False else _question_feedback(question, response, graded)
            if graded is False:
                print("Feedback: " + str(_question_feedback(question, response, graded)))
            # Record evidence only for explicitly covered topic IDs. Other topics have
            # no diagnostic result and consequently retain standard lesson depth.
            diag.setdefault("completed_question_indices", []).append(index)
            diag["attempts"] = int(diag.get("attempts", 0)) + 1
            diag.setdefault("questions", []).append({"index": index, "question": question["question"],
                "question_type": question["type"], "evidence_topics": [n["id"] for n in batch],
                "score": 1 if graded is True else (0 if graded is False else None)})
            save_profile(pfile, profile)
            for node in batch:
                record_event(profile, pfile, "pre_test", node["id"],
                    1 if graded is True else (0 if graded is False else None), misconception,
                    {"question": question["question"], "selected_option": response if isinstance(response, int) else None,
                     "response": response if isinstance(response, str) else None, "question_type": question["type"],
                     "diagnostic_question_index": index, "evidence_topics": [n["id"] for n in batch]})
            completed_indices.add(index)
        diag["completed"] = True
        diag["requested_question_count"] = len(batches)
        save_profile(pfile, profile)
        log("Diagnostic complete: %d scenario questions; all %d topics will still be taught." % (len(batches), len(nodes)))
    elif diag.get("completed"):
        log("Scenario diagnostic already completed; resuming saved progress.")

    groups = _chapter_groups(nodes)
    chapter_order = list(groups)
    # DAG traversal is deterministic in graph order; never skip based on diagnostic mastery.
    while len(mastered) < len(nodes):
        available = eligible_nodes(nodes, mastered)
        if not available:
            blocked = [n["id"] for n in nodes if n["id"] not in mastered]
            raise RuntimeError("No eligible topic remains; blocked topics: " + ", ".join(blocked))
        node = available[0]
        title = node.get("title", node["id"])
        fast = _mastery_level(profile, node["id"]) == "fast"
        log("Unlocked topic: %s | %s lesson" % (title, "fast-track" if fast else "scaffolded"))
        context = "Learner portfolio: %s. Evidence: %s." % (profile.get("portfolio", {}), "diagnostic suggests prior mastery" if fast else "build understanding step by step")
        lesson = provider.lesson(node, context, fast_track=fast)
        print("\nLesson: " + lesson["lesson"])
        print("Concept tip: " + lesson["tip"])
        record_event(profile, pfile, "micro_lesson", node["id"], None,
                     details={"lesson": lesson["lesson"], "tip": lesson["tip"], "track": "fast" if fast else "full"})
        question = provider.question(node, "lesson assessment quiz", profile)
        attempts = 0
        while True:
            response = get_choice(question, provider.simulate, True)
            graded = grade_answer(question, response)
            if graded is None and not provider.simulate:
                print("I cannot grade that wording confidently. " + question.get("explanation", ""))
                action = input("[R]etry / [S]elf-rate correct / [N]ot sure (pause): ").strip().lower()
                if action == "r":
                    response = get_choice(question, False)
                    graded = grade_answer(question, response)
                elif action == "s":
                    graded = True
                else:
                    log("Session paused; uncertain answer was not marked mastered.")
                    return 0
            misconception = None if graded is True else ("Answer could not be confidently graded." if graded is None else _question_feedback(question, response, graded))
            attempts += 1
            record_event(profile, pfile, "quiz_attempt", node["id"], 1 if graded is True else (0 if graded is False else None), misconception,
                         {"attempt": attempts, "question": question["question"], "selected_option": response if isinstance(response, int) else None,
                          "response": response if isinstance(response, str) else None, "question_type": question["type"]})
            _record_srs_review(scheduler, node["id"], graded, details={"source": "lesson_quiz", "attempt": attempts,
                                                                       "question_type": question["type"]})
            if graded is True:
                mastered.add(node["id"])
                profile["mastered_topics"] = sorted(mastered)
                save_profile(pfile, profile)
                log("Completed lesson: " + node["id"])
                break
            print("Feedback: " + str(_question_feedback(question, response, graded)))
            selected = question.get("options", [])[response] if isinstance(response, int) else response
            remediation = provider.remediation(node, question["question"], selected, misconception or "uncertain answer")
            print("Remediation: " + remediation)
            if provider.simulate:
                continue
            action = input("Enter R to retry, E for explanation, or Q to stop: ").strip().lower()
            if action == "q":
                log("Session paused. Completed %d/%d topics." % (len(mastered), len(nodes)))
                return 0
            if action == "e":
                print("Explanation: " + question["explanation"])
        # Only trigger after every lesson in the chapter is completed.
        chapter = _chapter_for(node)
        chapter_nodes = groups[chapter]
        if all(n["id"] in mastered for n in chapter_nodes):
            _chapter_quiz(provider, profile, pfile, chapter, chapter_nodes,
                          getattr(args, "chapter_quiz_questions", 3), scheduler)
    # Existing completed chapters might be pending quiz after resume/legacy profile.
    for chapter in chapter_order:
        if all(n["id"] in mastered for n in groups[chapter]):
            _chapter_quiz(provider, profile, pfile, chapter, groups[chapter], getattr(args, "chapter_quiz_questions", 3), scheduler)
    log("Course complete! All %d curriculum topics were taught and completed." % len(nodes))
    print("Mastered topics: " + ", ".join(n["id"] for n in nodes if n["id"] in mastered))
    return 0

def main(argv=None):
    parser = argparse.ArgumentParser(description="Adaptive runtime tutor for curriculum graph JSON.")
    parser.add_argument("--graph", default="curriculum_graph.json", help="Path to curriculum graph JSON.")
    parser.add_argument("--api-key", "-k", default=None, help="OpenRouter API key; otherwise OPENROUTER_API_KEY is used.")
    parser.add_argument("--model", default=None, help="OpenRouter model identifier (or OPENROUTER_MODEL in .env).")
    parser.add_argument("--simulate", action="store_true", help="Use deterministic mock content and answers.")
    parser.add_argument("--user-id", default=None, help="Learner identifier; saves to learner_profile_<user-id>.json unless --profile-file is set.")
    parser.add_argument("--profile-file", default=None, help="Explicit learner JSON profile path (overrides --user-id default).")
    parser.add_argument("--check-due-reviews", action="store_true", help="Print due FSRS reviews as JSON and exit.")
    parser.add_argument("--update-portfolio", action="store_true", help="Interactively edit/update the saved learner portfolio.")
    parser.add_argument("--diagnostic-question-count", type=int, default=4, help="Number of multi-topic scenario questions in the initial diagnostic (default: 4; clamped to 1–12 and the course topic count).")
    parser.add_argument("--chapter-quiz-questions", type=int, default=3, help="Questions per completed chapter quiz (default: 3).")
    args, _unknown = parser.parse_known_args(argv)
    try:
        return run(args)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        log("ERROR: " + str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
