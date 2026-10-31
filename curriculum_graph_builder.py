#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Generate and validate AI-powered curriculum dependency graphs.

The module uses the OpenRouter chat completions REST API directly. Set
OPENROUTER_API_KEY in the environment before making live requests.
"""

# ============================== Imports =====================================
import argparse
from difflib import SequenceMatcher
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import List, Literal, Tuple

import networkx as nx
import requests
from pydantic import BaseModel, Field, field_validator, model_validator

# ============================== Configuration ==============================
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
        # The .env file is optional; environment variables still work.
        pass


_load_dotenv()

MODEL = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 3
SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

# ============================== Data Schemas ===============================
class TopicNode(BaseModel):
    """One atomic, teachable topic in a curriculum."""

    id: str
    title: str
    description: str
    prerequisites: List[str] = Field(default_factory=list)
    bloom_level: Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"] = "Understand"
    estimated_minutes: int = 10
    # Optional for backwards compatibility with existing graph JSON files.
    module_id: str = ""
    module_title: str = ""
    lesson_index: int = 0
    # Optional chapter metadata; absent in older graph JSON remains valid.
    chapter_goal: str = ""
    subtopics: List[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_snake_case_id(cls, value):
        if not SNAKE_CASE_RE.fullmatch(value):
            raise ValueError("id must use lowercase snake_case English")
        return value


class CurriculumGraph(BaseModel):
    """Course metadata and its list of topic nodes."""

    course_name: str
    description: str
    nodes: List[TopicNode]

    @model_validator(mode="after")
    def validate_unique_ids(self):
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        return self


def _subtopic_covered(subtopic: str, lesson_text: str) -> bool:
    """Match a syllabus subtopic against lesson titles/details forgivingly.

    Punctuation and case are ignored; close spelling variants and well-known
    Stable Diffusion model aliases (SD/SDXL) are accepted as well. The fuzzy
    check compares phrase-sized word windows to avoid matching a long syllabus
    topic merely because one generic word (for example, "diffusion") appears.
    """
    def words(value):
        return re.findall(r"[a-z0-9]+", value.casefold())

    topic_words = words(subtopic)
    text_words = words(lesson_text)
    if not topic_words or not text_words:
        return False
    topic = "".join(topic_words)
    text_compact = "".join(text_words)
    if topic in text_compact:
        return True

    # Common shorthand/model names for the syllabus phrase.
    aliases = {
        "stablediffusion": {"sd", "sdxl"},
    }
    if topic in aliases and (aliases[topic].intersection(text_words) or "stablediffusionxl" in text_compact):
        return True

    # Catch small typos or tokenization variations while requiring a close
    # match to a phrase-sized window, not just a shared generic token.
    minimum = max(1, len(topic_words) - 1)
    maximum = min(len(text_words), len(topic_words) + 2)
    for size in range(minimum, maximum + 1):
        for offset in range(len(text_words) - size + 1):
            candidate = "".join(text_words[offset:offset + size])
            if abs(len(candidate) - len(topic)) <= max(2, int(len(topic) * 0.2)):
                if SequenceMatcher(None, topic, candidate).ratio() >= 0.84:
                    return True
    return False


# ============================== Builder ====================================
class CurriculumGraphBuilder:
    """Create, repair, save, load, and display curriculum graphs."""

    SYSTEM_PROMPT = """You are an expert instructional designer. Create a coherent course curriculum graph from the supplied syllabus chapters. A chapter is a parent container, not a lesson. Propose a pedagogically adequate VARIABLE number of atomic, assessable lessons per chapter based on its scope, goal, and parenthesized subtopics; never use a uniform default count. A narrow goal may need few lessons and a broad chapter more. If an optional explicit override count is present, follow it.
Every lesson must explicitly map to its input chapter through exact module_title and stable module_id, carry chapter_goal and subtopics unchanged, and state in its description which goal/subtopic(s) it teaches. This is an invariant: EVERY parenthesized/listed subtopic in every chapter MUST be explicitly mentioned or unambiguously mapped in at least one lesson title, description, or subtopics field (covered_subtopics). Do not omit, silently rename, or merely imply any subtopic. If combining topics in a lesson, explicitly name/map each one; minor spelling, casing, punctuation, or common model-alias variations are acceptable only when the intended mapping is unambiguous. Cover every chapter goal and all listed subtopics across its lessons. Preserve chapter order and titles.
Return strict JSON only, matching this runtime-compatible schema:
{"course_name":"string","description":"string","nodes":[{"id":"snake_case_english","title":"specific atomic lesson title","description":"teachable outcome with explicit goal/subtopic coverage","prerequisites":["existing_node_id"],"bloom_level":"Remember|Understand|Apply|Analyze|Evaluate|Create","estimated_minutes":10,"module_id":"stable chapter slug","module_title":"exact chapter title","lesson_index":1,"chapter_goal":"exact input goal or empty string","subtopics":["input subtopics"]}]}
Use unique IDs; number lesson_index from 1 within each chapter. Add only genuine prerequisite edges. Every prerequisite must reference an id in this response. Avoid cycles and redundant edges. Return no markdown."""

    def __init__(self, api_key=None, model=None):
        # Explicit CLI/API value wins; otherwise use the current environment/.env value.
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL") or MODEL

    def generate_graph(self, course_syllabus: str, lessons_per_heading: int = None) -> CurriculumGraph:
        """Generate chapter-mapped lessons; count is inferred unless explicitly overridden."""
        chapters, course_name = parse_syllabus(course_syllabus)
        if not chapters:
            raise ValueError("Course syllabus must contain at least one numbered chapter heading.")
        if lessons_per_heading is not None and lessons_per_heading < 1:
            raise ValueError("lessons_per_heading override must be at least 1.")
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not found. Set OPENROUTER_API_KEY in .env or the environment, or use --simulate.")
        payload = {
            "model": self.model or MODEL,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "course_name": course_name,
                    "chapters": chapters,
                    "lesson_count_override": lessons_per_heading,
                    "instructions": "Generate variable counts based on chapter scope; preserve metadata, demonstrate complete explicit coverage, and retain chapter order.",
                }, ensure_ascii=False)},
            ],
        }
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "Curriculum Graph Builder",
        }
        response = self._post_with_retries(headers, payload)
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned an unexpected response format.") from exc
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter response content was not a text JSON document.")
        graph = self._parse_graph_json(self._strip_markdown_fence(content))
        self._check_module_expansion(graph, chapters, lessons_per_heading)
        return graph

    @staticmethod
    def _check_module_expansion(graph, chapters, lessons_per_heading=None):
        """Require a lesson mapping and adequate explicit chapter-goal/subtopic coverage."""
        by_module = defaultdict(list)
        for node in graph.nodes:
            by_module[node.module_id].append(node)
        errors = []
        seen_modules = set()
        for chapter in chapters:
            module_id = slugify(chapter["title"])
            lessons = by_module.get(module_id, [])
            seen_modules.add(module_id)
            if not lessons or any(n.module_title.strip() != chapter["title"].strip() for n in lessons):
                errors.append("%s: missing lessons or exact module mapping" % chapter["title"])
                continue
            if lessons_per_heading is not None and len(lessons) != lessons_per_heading:
                errors.append("%s: expected %d lessons, got %d" % (chapter["title"], lessons_per_heading, len(lessons)))
            indices = sorted(n.lesson_index for n in lessons)
            if indices != list(range(1, len(lessons)+1)):
                errors.append("%s: lesson_index must be contiguous from 1" % chapter["title"])
            goals = " ".join((n.chapter_goal + " " + n.description).casefold() for n in lessons)
            if chapter["goal"] and chapter["goal"].casefold() not in goals:
                errors.append("%s: chapter goal not preserved/covered" % chapter["title"])
            covered = " ".join(
                (n.title + " " + n.description + " " + " ".join(n.subtopics or [])).casefold()
                for n in lessons
            )
            for subtopic in chapter["subtopics"]:
                if not _subtopic_covered(subtopic, covered):
                    errors.append("%s: subtopic not covered: %s" % (chapter["title"], subtopic))
        extra = set(by_module) - seen_modules
        if extra:
            errors.append("lessons assigned to unknown modules: " + ", ".join(sorted(extra)))
        if errors:
            raise ValueError("Model did not preserve complete chapter mapping/coverage: " + "; ".join(errors))

    def _post_with_retries(self, headers, payload):
        """Retry transient transport/server/rate-limit errors with exponential delay."""
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                # Guarantee OpenRouter receives a non-empty model even if a caller supplies an incomplete payload.
                payload = dict(payload)
                payload["model"] = payload.get("model") or MODEL
                response = requests.post(API_URL, headers=headers, json=payload,
                                         timeout=REQUEST_TIMEOUT_SECONDS)
                if response.status_code == 401:
                    raise RuntimeError("OpenRouter authentication failed (401). Check OPENROUTER_API_KEY.")
                if response.status_code == 429:
                    last_error = RuntimeError("OpenRouter rate limit reached (429).")
                    if attempt == MAX_ATTEMPTS - 1:
                        raise RuntimeError("OpenRouter rate limit reached (429) after 3 attempts.")
                elif response.status_code >= 500:
                    last_error = RuntimeError("OpenRouter server error (HTTP %s)." % response.status_code)
                    if attempt == MAX_ATTEMPTS - 1:
                        raise last_error
                else:
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as exc:
                        raise RuntimeError("OpenRouter request failed (HTTP %s): %s" %
                                           (response.status_code, response.text[:500])) from exc
                    return response
            except RuntimeError:
                # Authentication and final-attempt errors are actionable; do not retry them.
                if attempt == MAX_ATTEMPTS - 1 or "authentication failed" in str(sys.exc_info()[1]).lower():
                    raise
                last_error = sys.exc_info()[1]
            except requests.RequestException as exc:
                last_error = exc
                if attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError("OpenRouter request failed after 3 attempts: %s" % exc) from exc
            time.sleep(2 ** attempt)
        raise RuntimeError("OpenRouter request failed: %s" % last_error)

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        text = text.strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else text

    @staticmethod
    def _parse_graph_json(text: str) -> CurriculumGraph:
        try:
            if hasattr(CurriculumGraph, "model_validate_json"):
                return CurriculumGraph.model_validate_json(text)
            return CurriculumGraph.parse_raw(text)
        except Exception as exc:
            raise ValueError("Could not parse OpenRouter JSON as a CurriculumGraph: %s" % exc) from exc

    def validate_and_fix(self, graph: CurriculumGraph) -> Tuple[CurriculumGraph, List[str]]:
        """Repair dangling/self edges, break cycles, reduce edges, and verify a DAG."""
        if not isinstance(graph, CurriculumGraph):
            raise TypeError("graph must be a CurriculumGraph")
        fixes = []
        by_id = {node.id: node for node in graph.nodes}
        existing = set(by_id)
        prerequisites = {node.id: list(node.prerequisites) for node in graph.nodes}

        # Remove invalid references and self-loops before graph algorithms.
        for node_id in sorted(prerequisites):
            kept = []
            for prereq in prerequisites[node_id]:
                if prereq not in existing:
                    fixes.append("Dropped missing prerequisite '%s' from '%s'." % (prereq, node_id))
                elif prereq == node_id:
                    fixes.append("Removed self-loop on '%s'." % node_id)
                elif prereq not in kept:
                    kept.append(prereq)
            prerequisites[node_id] = kept

        # Edge direction is prerequisite -> dependent, as expected by topological order.
        graph_nx = nx.DiGraph()
        graph_nx.add_nodes_from(sorted(existing))
        for target, prereqs in prerequisites.items():
            for source in prereqs:
                graph_nx.add_edge(source, target)

        # Break one weakest edge from a current cycle, then recompute cycles.
        while not nx.is_directed_acyclic_graph(graph_nx):
            cycles = list(nx.simple_cycles(graph_nx))
            if not cycles:
                raise RuntimeError("Cycle detection reported a non-DAG but found no cycle.")
            cycle = min((tuple(c) for c in cycles), key=lambda c: tuple(sorted(c)))
            cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
            source, target = min(cycle_edges, key=lambda edge: (graph_nx.in_degree(edge[1]), edge[1], edge[0]))
            graph_nx.remove_edge(source, target)
            prerequisites[target].remove(source)
            fixes.append("Broke cycle by removing prerequisite '%s' from '%s' (target had %d prerequisites)." %
                         (source, target, graph_nx.in_degree(target)))

        if not nx.is_directed_acyclic_graph(graph_nx):
            raise RuntimeError("Internal validation error: graph is not acyclic after cycle repair.")
        fixes.append("DAG verified: graph is acyclic.")

        reduced = nx.transitive_reduction(graph_nx)
        removed = sorted(set(graph_nx.edges()) - set(reduced.edges()))
        if removed:
            for source, target in removed:
                prerequisites[target].remove(source)
            fixes.append("Transitive reduction removed %d redundant edge(s): %s." %
                         (len(removed), ", ".join("%s -> %s" % edge for edge in removed)))
        else:
            fixes.append("Transitive reduction removed 0 redundant edges.")

        updated_nodes = []
        for node in graph.nodes:
            copy_method = getattr(node, "model_copy", None)
            updated_nodes.append(copy_method(update={"prerequisites": prerequisites[node.id]})
                                 if copy_method else node.copy(update={"prerequisites": prerequisites[node.id]}))
        graph_copy = getattr(graph, "model_copy", None)
        fixed = graph_copy(update={"nodes": updated_nodes}) if graph_copy else graph.copy(update={"nodes": updated_nodes})
        # Ensure current graph and reported graph represent identical edges.
        if not nx.is_directed_acyclic_graph(self._to_networkx(fixed)):
            raise RuntimeError("Internal validation error: final graph is not a DAG.")
        return fixed, fixes

    @staticmethod
    def _to_networkx(graph: CurriculumGraph) -> nx.DiGraph:
        result = nx.DiGraph()
        result.add_nodes_from(node.id for node in graph.nodes)
        for node in graph.nodes:
            for prereq in node.prerequisites:
                result.add_edge(prereq, node.id)
        return result

    def topological_order(self, graph: CurriculumGraph) -> List[str]:
        """Return topic ids in deterministic prerequisite-first order."""
        return list(nx.lexicographical_topological_sort(self._to_networkx(graph), key=str))

    def save(self, graph: CurriculumGraph, path="curriculum_graph.json") -> str:
        """Write the graph as indented UTF-8 JSON and return the output path."""
        with open(path, "w", encoding="utf-8") as handle:
            if hasattr(graph, "model_dump_json"):
                handle.write(graph.model_dump_json(indent=2))
            else:
                handle.write(graph.json(indent=2))
            handle.write("\n")
        return path

    @staticmethod
    def load(path="curriculum_graph.json") -> CurriculumGraph:
        """Load and validate a graph JSON file."""
        with open(path, "r", encoding="utf-8") as handle:
            return CurriculumGraphBuilder._parse_graph_json(handle.read())

    def visualize(self, graph: CurriculumGraph) -> str:
        """Return a plain-ASCII view grouping nodes by longest prerequisite depth."""
        nx_graph = self._to_networkx(graph)
        if not nx.is_directed_acyclic_graph(nx_graph):
            raise ValueError("Cannot visualize a cyclic graph; call validate_and_fix first.")
        depths = {}
        for node_id in nx.topological_sort(nx_graph):
            depths[node_id] = max((depths[parent] + 1 for parent in nx_graph.predecessors(node_id)), default=0)
        nodes_by_id = {node.id: node for node in graph.nodes}
        grouped = defaultdict(list)
        for node_id, depth in depths.items():
            grouped[depth].append(node_id)
        lines = ["Curriculum levels (ASCII; prerequisites appear at earlier levels):"]
        for depth in sorted(grouped):
            lines.append("Level %d:" % depth)
            for node_id in sorted(grouped[depth]):
                node = nodes_by_id[node_id]
                prereq_text = ", ".join(sorted(node.prerequisites)) if node.prerequisites else "none"
                lines.append("  - %s: %s [prerequisites: %s]" % (node_id, node.title, prereq_text))
        return "\n".join(lines)

# ============================== Simulation =================================
def slugify(text: str) -> str:
    """Stable ASCII snake-case identifier for a chapter title."""
    value = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    return value or "chapter"


def _split_heading_goal(text: str):
    """Split on the first colon outside parentheses, preserving internal colons."""
    depth = 0
    for i, char in enumerate(text):
        if char == "(": depth += 1
        elif char == ")": depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return text[:i].strip(), text[i+1:].strip()
    return text.strip(), ""


def parse_syllabus(syllabus: str):
    """Parse Course metadata and numbered chapters with goals/subtopics."""
    if not isinstance(syllabus, str) or not syllabus.strip():
        return [], ""
    course_name = ""
    chapters = []
    for raw in syllabus.splitlines():
        line = re.sub(r"^\s*#{1,6}\s*", "", raw).strip()
        if not line:
            continue
        course_match = re.match(r"^Course\s*:\s*(.*)$", line, re.I)
        if course_match:
            course_name = course_match.group(1).strip()
            continue
        match = re.match(r"^(?:[-*+]\s*)?(\d+)[.)]\s*(.+)$", line)
        if not match:
            continue
        title, goal = _split_heading_goal(match.group(2).strip())
        subtopics = []
        for group in re.findall(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", title):
            subtopics.extend(part.strip() for part in re.split(r",|;", group) if part.strip())
        chapters.append({"number": int(match.group(1)), "title": title,
                         "goal": goal, "subtopics": subtopics,
                         "module_id": slugify(title)})
    return chapters, course_name


def parse_syllabus_headings(syllabus: str) -> List[str]:
    """Backward-compatible helper returning chapter titles only."""
    chapters, _course = parse_syllabus(syllabus)
    if chapters:
        return [chapter["title"] for chapter in chapters]
    # Legacy unnumbered outlines remain accepted by this helper.
    result = []
    for raw in (syllabus or "").splitlines():
        line = re.sub(r"^\s*#{1,6}\s*", "", raw).strip()
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
        if line and not re.match(r"^Course\s*:", line, re.I):
            result.append(_split_heading_goal(line)[0])
    return result


def build_simulated_graph(lessons_per_heading: int = None, syllabus: str = None) -> CurriculumGraph:
    """Build an offline graph from real chapter scope; optional count override is legacy."""
    if syllabus:
        chapters, course_name = parse_syllabus(syllabus)
        if not chapters:
            raise ValueError("Simulation syllabus must contain numbered chapter headings.")
    else:
        syllabus = """Course: AI Tools
1. AI Fundamentals: Understand core AI concepts and limitations
2. Text Generation (Prompting, Tokens): Create and assess generated text
3. Multimedia Tools (Image Generation, Audio Generation, Evaluation): Compare generated media"""
        chapters, course_name = parse_syllabus(syllabus)
    if lessons_per_heading is not None and lessons_per_heading < 1:
        raise ValueError("lessons_per_heading override must be at least 1.")

    nodes = []
    previous_chapter_last = None
    for chapter in chapters:
        # Scope proxies are the actual parenthesized concepts and distinct title areas.
        title_parts = [part.strip() for part in re.split(r"\s+(?:&|and)\s+", re.sub(r"\([^)]*\)", "", chapter["title"]), flags=re.I) if part.strip()]
        concepts = chapter["subtopics"] or title_parts or [chapter["title"]]
        count = lessons_per_heading if lessons_per_heading is not None else len(concepts)
        count = max(1, min(count, len(concepts))) if lessons_per_heading is not None else count
        # When multiple atomic concepts are listed, one lesson focuses on each.
        lesson_concepts = concepts[:count]
        if len(lesson_concepts) < count:
            lesson_concepts = lesson_concepts + [concepts[-1]] * (count-len(lesson_concepts))
        module_id = chapter["module_id"]
        chapter_nodes = []
        for index, concept in enumerate(lesson_concepts, 1):
            lesson_id = "%s_lesson_%02d" % (module_id, index)
            label = concept.strip().rstrip(".")
            title = ("Foundations of " + label) if index == 1 and len(lesson_concepts) == 1 else ("Apply " + label if index == 1 else "Practice " + label)
            goal_text = chapter["goal"] or "Develop understanding of " + chapter["title"]
            description = "Learn and apply %s to support chapter goal: %s" % (label, goal_text)
            if chapter["subtopics"]:
                description += "; mapped subtopic: %s" % label
            prereqs = [chapter_nodes[-1].id] if chapter_nodes else ([previous_chapter_last] if previous_chapter_last else [])
            node = TopicNode(id=lesson_id, title=title, description=description,
                             prerequisites=prereqs, bloom_level="Apply" if index > 1 else "Understand",
                             estimated_minutes=15, module_id=module_id, module_title=chapter["title"],
                             lesson_index=index, chapter_goal=chapter["goal"], subtopics=chapter["subtopics"])
            chapter_nodes.append(node)
        nodes.extend(chapter_nodes)
        previous_chapter_last = chapter_nodes[-1].id
    return CurriculumGraph(course_name=course_name or "Course", description="Offline scope-based curriculum graph.", nodes=nodes)

# ============================== CLI ========================================
def _read_syllabus(value):
    if value and os.path.isfile(value) and value.lower().endswith(".txt"):
        with open(value, "r", encoding="utf-8") as handle:
            return handle.read()
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and validate a curriculum dependency graph.")
    parser.add_argument("--syllabus", help="Syllabus text or path to a .txt file.")
    parser.add_argument("--out", default="curriculum_graph.json", help="Output JSON path.")
    parser.add_argument("--model", default=None, help="OpenRouter model id (or OPENROUTER_MODEL in .env).")
    parser.add_argument("--api-key", "-k", dest="api_key", default=None,
                        help="OpenRouter API key (overrides OPENROUTER_API_KEY and .env).")
    parser.add_argument("--lessons-per-heading", type=int, default=None, metavar="N",
                        help="Optional legacy override: exact lesson count per chapter (default: auto based on scope).")
    parser.add_argument("--simulate", action="store_true", help="Run the offline multi-lesson-per-module example.")
    args, _unknown = parser.parse_known_args(argv)  # Safe in notebooks and Colab.
    if args.lessons_per_heading is not None and args.lessons_per_heading < 1:
        parser.error("--lessons-per-heading must be at least 1.")

    builder = CurriculumGraphBuilder(api_key=args.api_key, model=args.model)
    try:
        if args.simulate:
            raw_graph = build_simulated_graph(args.lessons_per_heading, _read_syllabus(args.syllabus) if args.syllabus else None)
            print("Simulation mode: using supplied syllabus scope." if args.syllabus else "Simulation mode: using built-in example.")
        else:
            if not args.syllabus:
                parser.error("--syllabus is required unless --simulate is used.")
            syllabus = _read_syllabus(args.syllabus)
            raw_graph = builder.generate_graph(syllabus, args.lessons_per_heading)
        fixed_graph, fixes = builder.validate_and_fix(raw_graph)
        output_path = builder.save(fixed_graph, args.out)
        order = builder.topological_order(fixed_graph)
        edge_count = sum(len(node.prerequisites) for node in fixed_graph.nodes)

        print("\nFinal summary")
        print("Course: %s" % fixed_graph.course_name)
        print("Nodes: %d" % len(fixed_graph.nodes))
        print("Edges: %d" % edge_count)
        print("Fixes and validation:")
        for fix in fixes:
            print("  - %s" % fix)
        print("Topological learning order:")
        print("  " + " -> ".join(order))
        print(builder.visualize(fixed_graph))
        print("Saved JSON: %s" % output_path)
        return 0
    except Exception as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
