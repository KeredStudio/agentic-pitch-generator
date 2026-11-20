# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
/**
 * @file agent.py
 * @description Multi-agent ADK workflow definition and service-injected execution
 *   engine for the Agentic Pitch Generator.
 *
 * Why: Defines the Google ADK specialist agent team (`creative_director`,
 * `brand_strategist`, `copywriter`, `visual_director`), structured payload contracts,
 * `LoopGuard` DAG validation, `assemble = JoinNode(name="assemble")`, `package`,
 * `root_agent`, and `app` running on Gemini Enterprise Agent Platform (`gemini-3.8-flash`)
 * while providing lightweight stdlib fallback shims when `google-adk` / `google-genai`
 * wheels are not installed locally. Also exposes `run_pitch_workflow` and
 * `resume_pitch_workflow` wired through `ServiceContainer` so local runs use live
 * Cloud models and offline test suites execute deterministically.
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/agent.py`)
#   - Step 1a: `# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]`
#   - Step 1b: `# [Guidepost — Step 1b: Skill Evaluation Harness]`
#   - Step 1c: `# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]`
#   - Step 3a: `# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]`
#   - Step 3c: `# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]`
#   - Step 4a: `# [Guidepost — Step 4a: Tokenomics & Context Optimization]`
# (Tip: All specialist agents, workflow edges, and lab guideposts are in the top
#  ~320 lines of this file—read lines 1–350 in a single `view_file` call!)
# ==============================================================================

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import time
from typing import Any

from pitch_generator.app_utils.adk_primitives import (
    Agent,
    App,
    ArtDirectionPayload,
    CampaignConcept,
    CircularLoopError,
    ConceptPayload,
    Context,
    CopyPayload,
    Event,
    Gemini,
    JoinNode,
    LoopGuard,
    PayloadValidationError,
    PitchPackage,
    RemoteA2aAgent,
    RequestInput,
    VisualPayload,
    Workflow,
    _DualSyncAsyncEventStream,
    _GenAITypesShim,
    _SessionData,
    _extract_json_Candidate,
    _key_visual,
    _node_name,
    _normalize_directed_edges,
    extract_json_payload,
    node,
    parse_json_payload,
    parse_payload,
    strip_markdown_fences,
    types,
)
from pitch_generator.app_utils.services import (
    ServiceContainer,
    get_default_services,
)
from pitch_generator.config import (
    PitchConfig,
    get_config,
)

MODEL: str = get_config().flash_model
IMAGE_MODEL: str = get_config().image_model


def _model(config: PitchConfig | None = None) -> Gemini:
    """
    /**
     * Construct the default `Gemini` model configuration (`gemini-3.8-flash`) for
     * specialist agents.
     *
     * Why: Centralizes `location="global"` and `HttpRetryOptions(attempts=3)` so every
     * specialist agent shares resilient retry settings on Gemini Enterprise Agent Platform.
     *
     * @param config Optional `PitchConfig` override.
     * @return Configured `Gemini` model descriptor.
     */
    """
    cfg = config or get_config()
    return Gemini(
        model=cfg.flash_model,
        client_kwargs={"location": cfg.location},
        retry_options=types.HttpRetryOptions(attempts=3),
    )


_build_model = _model


async def generate_key_visual(
    art_direction: str,
    tool_context: Context | None = None,
    *,
    simulate_no_image: bool = False,
) -> dict[str, Any]:
    """
    /**
     * Generate a 16:9 key visual PNG from art direction notes and save it to session artifacts.
     *
     * Why: Attached to `visual_director` (`tools=[generate_key_visual]`) so the Visual
     * Director specialist can synthesize a key visual image via the configured
     * `image_model` (`gemini-nano-banana-2.1`) and store it in the session artifact store.
     *
     * @param art_direction Art direction prompt describing palette, lighting, composition, and subject.
     * @param tool_context Optional ADK `Context` / `ToolContext` for saving the image artifact.
     * @param simulate_no_image If True, simulates an empty image response to test error handling.
     * @return Metadata dictionary with `filename`, `version`, `bytes`, and `mime_type`.
     */
    """
    import mimetypes

    ctx = tool_context if tool_context is not None else Context()
    cfg = ctx.services.config if hasattr(ctx, "services") and ctx.services else get_config()
    if (
        simulate_no_image
        or getattr(ctx, "simulate_no_image", False)
        or not isinstance(art_direction, str)
        or not art_direction.strip()
    ):
        raise ValueError(f"{cfg.image_model} returned no image for: {str(art_direction)[:120]}")

    image_bytes, mime_type = ctx.services.llm.generate_image(
        art_direction.strip(),
        model=cfg.image_model,
    )
    if not image_bytes:
        raise ValueError(f"{cfg.image_model} returned no image for: {art_direction[:120]}")

    ext = mimetypes.guess_extension(mime_type or "") or ".png"
    filename = f"key_visual{ext}"
    image_part = types.Part(
        inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
    )
    version = await ctx.save_artifact(filename, image_part)
    ctx.session.events.append(
        Event(
            author="visual_director",
            content=types.Content(role="model", parts=[image_part]),
        )
    )
    return {
        "filename": filename,
        "version": version,
        "bytes": len(image_bytes),
        "mime_type": mime_type,
    }


def build_specialist_team(config: PitchConfig | None = None) -> dict[str, Agent]:
    """
    /**
     * Instantiate the four domain specialist agents for the Pitch Generator team.
     *
     * Why: Provides a deterministic factory that binds each specialist role
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) to
     * a focused single-responsibility instruction and a unique `output_key` so
     * downstream graph nodes can read each specialist's output without collision.
     *
     * @param config Optional `PitchConfig` used to resolve model settings.
     * @return Dictionary mapping role name to its configured `Agent` instance.
     */
    """
    model = _model(config)
    return {
        "creative_director": Agent(
            name="creative_director",
            model=model,
            description="Turns a raw campaign topic into a punchy concept line and strategic rationale.",
            instruction=(
                "You are the Creative Director. Turn the campaign idea you are given into "
                "ONE punchy campaign concept line accompanied by a clear rationale explanation."
            ),
            output_key="creative_director",
        ),
        "copywriter": Agent(
            name="copywriter",
            model=model,
            description="Writes a concise, high-impact social media caption under 25 words.",
            instruction=(
                "You are the Copywriter. Write ONE short social caption for the campaign "
                "concept you are given. Strictly under 25 words with an energetic call to action."
            ),
            output_key="copywriter",
        ),
        "brand_strategist": Agent(
            name="brand_strategist",
            model=model,
            description="Defines campaign positioning, target audience, and brand tone for the campaign concept.",
            instruction=(
                "You are the Brand Strategist. Define the campaign positioning, "
                "target audience, and brand tone for the campaign concept."
            ),
            output_key="brand_strategist",
        ),
        "visual_director": Agent(
            name="visual_director",
            model=model,
            description="Translates the campaign concept into art direction and key visuals.",
            instruction="You are the Visual Director.",
            tools=[generate_key_visual],
            output_key="visual_director",
        ),
    }


create_specialist_agents = build_specialist_team
get_specialist_agents = build_specialist_team

SPECIALIST_AGENTS: dict[str, Agent] = build_specialist_team()

creative_director: Agent = SPECIALIST_AGENTS["creative_director"]
copywriter: Agent = SPECIALIST_AGENTS["copywriter"]
brand_strategist: Agent = SPECIALIST_AGENTS["brand_strategist"]
visual_director: Agent = SPECIALIST_AGENTS["visual_director"]


def get_specialist_agent(role: str) -> Agent:
    """
    /**
     * Look up a single specialist `Agent` by role identifier.
     *
     * Why: Fails fast with a descriptive `KeyError` when an invalid or unknown
     * specialist role is requested instead of silently returning `None`.
     *
     * @param role Specialist role name (e.g., `"creative_director"`, `"copywriter"`).
     * @return Matching `Agent` instance.
     */
    """
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Specialist role name must be a non-empty string")
    clean_role = role.strip()
    if clean_role == "market_researcher":
        return SPECIALIST_AGENTS["brand_strategist"]
    if clean_role not in SPECIALIST_AGENTS:
        raise KeyError(
            f"Unknown specialist role {role!r}. Expected one of {tuple(SPECIALIST_AGENTS.keys())}."
        )
    return SPECIALIST_AGENTS[clean_role]


def run_specialist_team(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    config: PitchConfig | None = None,
) -> dict[str, str]:
    """
    /**
     * Execute all four specialist agents sequentially/in-parallel against a campaign brief.
     *
     * Why: Allows learners and automated tests to verify that all four specialist roles
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) generate
     * domain-appropriate outputs offline via dependency-injected `ServiceContainer`.
     *
     * @param brief Non-empty campaign topic brief.
     * @param services Optional injected `ServiceContainer` for offline LLM execution.
     * @param config Optional `PitchConfig` override.
     * @return Dictionary mapping each specialist's `output_key` to its generated text.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string")

    clean_brief = brief.strip()
    active_services = services or get_default_services(config)
    team = build_specialist_team(active_services.config)
    model_id = active_services.config.flash_model

    concept_out = active_services.llm.generate_text(
        clean_brief,
        system_instruction=team["creative_director"].instruction,
        model=model_id,
    )
    copy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["copywriter"].instruction,
        model=model_id,
    )
    copy_words = copy_out.split()
    if len(copy_words) > 25:
        copy_out = " ".join(copy_words[:24])

    strategy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["brand_strategist"].instruction,
        model=model_id,
    )
    visual_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["visual_director"].instruction,
        model=model_id,
    )

    return {
        "creative_director": concept_out,
        "copywriter": copy_out,
        "brand_strategist": strategy_out,
        "visual_director": visual_out,
    }


execute_specialists = run_specialist_team
run_specialists = run_specialist_team

# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]
# TODO (Step 1a): Create the house brand skill and attach it to `visual_director`:
#   1. Create `pitch_generator/skills/brand-guidelines/SKILL.md` with YAML frontmatter
#      (`name: brand-guidelines`, `description: ... Use when ...`) and five Markdown sections:
#      `## Palette` (deep indigo, warm amber, slate, terracotta),
#      `## Light` (warm raking golden-hour light with long soft shadows),
#      `## Composition` (off-center hero subject, generous negative space, shallow depth of field),
#      `## Subject` (one realistic photographic subject, authentic/handcrafted materials), and
#      `## Never` (no neon, fluorescent cyan/magenta, watermarks, logos, ring light, or 3D renders).
#      Keep these 5 sections, but set values to match learner's prompt incase they get creative.
#   2. Define `Skill(name, description, body, path=None)` and `SkillToolset(skills)` (with
#      `.load_skill(skill_name) -> str` and `.loaded_skills: list[str]`), plus
#      `load_skill_from_dir(skill_dir: str | Path) -> Skill` and `load_brand_skill() -> Skill`.
#   3. Load `brand_guidelines_skill = load_skill_from_dir(...)` from
#      `pitch_generator/skills/brand-guidelines`, attach
#      `tools=[SkillToolset([brand_guidelines_skill]), generate_key_visual]` to `visual_director`,
#      and update `visual_director.instruction` to first call `load_skill("brand-guidelines")`,
#      write the art direction, call `generate_key_visual`, and output ONLY the art direction text.


class Skill:
    """
    /**
     * Agent Skill domain instructions loaded on-demand from structured Markdown.
     *
     * Why: Encapsulates domain expertise into an external modular skill definition
     * with lightweight YAML metadata at startup and on-demand body loading.
     */
    """

    def __init__(
        self,
        name: str,
        description: str,
        body: str,
        path: Path | str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.body = body
        self.path = Path(path) if path is not None else None

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, description={self.description[:40]!r}...)"


class SkillToolset:
    """
    /**
     * ADK-compatible toolset exposing on-demand skill retrieval to an agent.
     *
     * Why: Intercepts `load_skill` tool calls, records activated skill names in
     * `loaded_skills` for compliance evaluation, and returns the full skill body.
     */
    """

    def __init__(self, skills: Sequence[Skill]) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in skills}
        self.loaded_skills: list[str] = []

    def load_skill(self, skill_name: str) -> str:
        """
        /**
         * Load a registered skill by name.
         *
         * @param skill_name The identifier of the skill to load.
         * @return The markdown instructions of the skill.
         */
        """
        if skill_name in self._skills:
            if skill_name not in self.loaded_skills:
                self.loaded_skills.append(skill_name)
            return self._skills[skill_name].body
        return f"Skill '{skill_name}' not found. Available skills: {list(self._skills.keys())}"


def load_skill_from_dir(skill_dir: str | Path) -> Skill:
    """
    /**
     * Parse a SKILL.md file from a skill directory into a Skill instance.
     *
     * Why: Reads the YAML frontmatter (`name`, `description`) and body content
     * from the standard SKILL.md format.
     */
    """
    path = Path(skill_dir)
    skill_file = path / "SKILL.md" if path.is_dir() else path
    if not skill_file.is_file():
        raise FileNotFoundError(f"Skill file not found at {skill_file}")

    content = skill_file.read_text(encoding="utf-8")
    name = path.name if path.is_dir() else path.stem
    description = ""
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()
            for line in fm_text.splitlines():
                line = line.strip()
                if line.startswith("name:"):
                    name = line.split("name:", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("description:"):
                    description = line.split("description:", 1)[1].strip().strip('"').strip("'")

    return Skill(name=name, description=description, body=body, path=skill_file)


def load_brand_skill() -> Skill:
    """
    /**
     * Load the canonical brand guidelines skill from the package skills directory.
     *
     * Why: Provides a single helper to load `brand-guidelines` for visual director.
     */
    """
    skill_dir = Path(__file__).resolve().parent / "skills" / "brand-guidelines"
    return load_skill_from_dir(skill_dir)


brand_guidelines_skill = load_brand_skill()
brand_skill_toolset = SkillToolset([brand_guidelines_skill])

# Wire SkillToolset and instruction to visual_director in SPECIALIST_AGENTS
visual_director.tools = [brand_skill_toolset, generate_key_visual]
visual_director.instruction = (
    "You are the Visual Director. First call load_skill(\"brand-guidelines\") to load "
    "our house brand aesthetic rules. Then, based on the campaign concept, write "
    "brand-compliant art direction notes strictly adhering to our house palette, lighting, "
    "composition, and subject rules without ever mentioning or quoting any forbidden or excluded words. "
    "Finally call generate_key_visual with your art direction text. Output ONLY the art direction text."
)


# [Guidepost — Step 1b: Skill Evaluation Harness]
# TODO (Step 1b): Build a deterministic evaluation harness to verify skill activation and brand compliance:
#   - `FORBIDDEN_BRAND_PATTERNS: dict[str, str]`: regex patterns for forbidden visual styles
#     (`neon`, `fluorescent`, `cyan`, `magenta`, `watermark`, `logo`, `3d_render`, `ring_light`).
#   - `SkillEvalResult` dataclass with fields: `score: float`, `passed: bool`, `loaded_skill: bool`,
#     `violations: list[str]`, and `pillar_scores: dict[str, float]`.
#   - `evaluate_brand_skill(art_direction: str, loaded_skills: Sequence[str] = ("brand-guidelines",)) -> SkillEvalResult`:
#     verifies `"brand-guidelines"` is in `loaded_skills`, scores the 4 positive pillars
#     (`palette`, `light`, `composition`, `subject` at `0.25` each), deducts `0.25` per forbidden match,
#     and sets `passed = bool(loaded_skill and not violations and score >= 0.75)`.
#   - `run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]`:
#     runs compliant, off-brand, and unloaded-skill benchmark cases and returns a summary dict with
#     `"total"`, `"passed"`, `"results"`, and `"all_expectations_met"`.

FORBIDDEN_BRAND_PATTERNS: dict[str, str] = {
    "neon": r"\bneon\b",
    "fluorescent": r"\bfluorescent\b",
    "cyan": r"\bcyan\b",
    "magenta": r"\bmagenta\b",
    "watermark": r"\bwatermark(s)?\b",
    "logo": r"\blogo(s)?\b",
    "3d_render": r"\b(3d\s+render|cgi|octane\s+render|photorealistic\s+3d)\b",
    "ring_light": r"\bring\s+light\b",
}

POSITIVE_BRAND_PILLARS: dict[str, list[str]] = {
    "palette": ["indigo", "amber", "slate", "terracotta"],
    "light": ["golden hour", "raking", "soft shadow", "warm light", "warm raking"],
    "composition": ["off-center", "negative space", "shallow depth of field", "depth of field"],
    "subject": ["photographic", "handcrafted", "authentic", "realistic"],
}


@dataclass
class SkillEvalResult:
    """
    /**
     * Evaluation result for brand guidelines compliance.
     *
     * Why: Provides a structured deterministic metric object capturing skill activation,
     * positive brand pillars scored, forbidden term violations, and final pass/fail status.
     */
    """

    score: float
    passed: bool
    loaded_skill: bool
    violations: list[str] = field(default_factory=list)
    pillar_scores: dict[str, float] = field(default_factory=dict)


def evaluate_brand_skill(
    art_direction: str,
    loaded_skills: Sequence[str] = ("brand-guidelines",),
) -> SkillEvalResult:
    """
    /**
     * Deterministically evaluate art direction text against house brand guidelines.
     *
     * Why: Ensures visual direction adheres to the four brand pillars while avoiding
     * off-brand forbidden terms (neon, 3D renders, watermarks).
     *
     * @param art_direction Generated art direction text.
     * @param loaded_skills Sequence of skill names recorded during generation.
     * @return SkillEvalResult with deterministic score and violation breakdown.
     */
    """
    text_lower = (art_direction or "").lower()
    has_loaded_skill = "brand-guidelines" in loaded_skills

    pillar_scores: dict[str, float] = {}
    total_pillar_score = 0.0

    for pillar, terms in POSITIVE_BRAND_PILLARS.items():
        matched = any(term in text_lower for term in terms)
        p_score = 0.25 if matched else 0.0
        pillar_scores[pillar] = p_score
        total_pillar_score += p_score

    violations: list[str] = []
    penalty = 0.0
    for term_name, pattern in FORBIDDEN_BRAND_PATTERNS.items():
        if re.search(pattern, text_lower):
            violations.append(term_name)
            penalty += 0.25

    final_score = max(0.0, total_pillar_score - penalty)
    passed = bool(has_loaded_skill and not violations and final_score >= 0.75)

    return SkillEvalResult(
        score=round(final_score, 2),
        passed=passed,
        loaded_skill=has_loaded_skill,
        violations=violations,
        pillar_scores=pillar_scores,
    )


def run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    /**
     * Run evaluation suite across benchmark art direction test cases.
     *
     * Why: Verifies that brand compliance grading performs as expected across compliant,
     * off-brand, and unloaded-skill test vectors.
     */
    """
    if cases is None:
        cases = [
            {
                "name": "compliant_house_style",
                "text": (
                    "Photographic authentic subject framed off-center with generous negative space "
                    "and shallow depth of field. Deep indigo and slate background with warm amber "
                    "raking golden-hour light casting long soft shadows."
                ),
                "loaded_skills": ["brand-guidelines"],
                "expected_pass": True,
            },
            {
                "name": "forbidden_neon_3d",
                "text": "A glossy 3D render lit by fluorescent cyan and neon magenta ring light with a giant logo watermark.",
                "loaded_skills": ["brand-guidelines"],
                "expected_pass": False,
            },
            {
                "name": "unloaded_skill",
                "text": "Authentic handcrafted subject in warm golden hour light with indigo tones.",
                "loaded_skills": [],
                "expected_pass": False,
            },
        ]

    results: list[dict[str, Any]] = []
    all_expectations_met = True

    for c in cases:
        eval_res = evaluate_brand_skill(
            art_direction=c.get("text", ""),
            loaded_skills=c.get("loaded_skills", ["brand-guidelines"]),
        )
        passed = eval_res.passed
        expected = c.get("expected_pass")
        matches_expectation = (passed == expected) if expected is not None else True
        if not matches_expectation:
            all_expectations_met = False

        results.append({
            "name": c.get("name", "unnamed"),
            "score": eval_res.score,
            "passed": eval_res.passed,
            "loaded_skill": eval_res.loaded_skill,
            "violations": eval_res.violations,
            "matches_expectation": matches_expectation,
        })

    return {
        "total": len(cases),
        "passed": sum(1 for r in results if r["passed"]),
        "results": results,
        "all_expectations_met": all_expectations_met,
    }


# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]
# TODO (Step 1c): Expose `visual_director` over the Agent2Agent (A2A) protocol and wire `RemoteA2aAgent`:
#   - `AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"`, `AgentCardBuilder(agent, rpc_url)`,
#     and `build_visual_director_card(rpc_url: str = "http://localhost:8801/a2a/visual_director") -> dict[str, Any]`
#     returning an A2A Agent Card with `"name": "visual_director"`, `"url": rpc_url`, and `"preferredTransport": "JSONRPC"`.
#   - `A2aAgentExecutorConfig`, `include_artifacts_in_a2a_event_interceptor`, and
#     `build_a2a_visual_director_app(services: ServiceContainer | None = None, rpc_url: str = ...) -> Any`
#     configured with `execute_interceptors=[include_artifacts_in_a2a_event_interceptor]` (also exposed in
#     `pitch_generator/fast_api_app.py` when `SERVICE_ROLE == "visual-director"`).
#   - `_cloud_run_client(base_url: str)`: returns `None` for local `http://` or empty URLs, and returns an
#     authenticated HTTP client with `timeout=600.0` and an OIDC `Authorization: Bearer <token>` header for `https://` Cloud Run URLs.
#   - `_pitch_parts_only(part: Any)`: stream filter that drops `function_call` and `function_response` parts
#     (returning `None` / `False`) while keeping `text` and `inline_data` (`image/png`) parts.
#   - `remote_visual_director = RemoteA2aAgent(name="visual_director", description=..., agent_card=f"{visual_director_url}/.well-known/agent-card.json", httpx_client=_cloud_run_client(visual_director_url), genai_part_converter=_pitch_parts_only, output_key="visual_director")`
#     and swap `visual_director` for `remote_visual_director` in `root_agent.edges`. Also remove `hidden`
#     from `#btn-inspect-a2a` in `frontend/index.html` (defining `remote_visual_director` or
#     `build_a2a_visual_director_app` also automatically sets `a2a_enabled=True` in `/api/config`).

AGENT_CARD_WELL_KNOWN_PATH: str = "/.well-known/agent-card.json"


def build_visual_director_card(
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> dict[str, Any]:
    """
    /**
     * Build the A2A Agent Card dictionary for the standalone Visual Director service.
     *
     * Why: Publishes the capability contract of the Visual Director specialist agent
     * over A2A protocol v0.3.
     */
    """
    from pitch_generator.app_utils.a2a import AgentCardBuilder

    builder = AgentCardBuilder(agent=visual_director, rpc_url=rpc_url)
    card = builder.build()
    card["name"] = "visual_director"
    card["url"] = rpc_url
    card["preferredTransport"] = "JSONRPC"
    return dict(card)


def build_a2a_visual_director_app(
    services: ServiceContainer | None = None,
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> Any:
    """
    /**
     * Build standalone ASGI app exposing the Visual Director microservice over A2A.
     *
     * Why: Enables independent scaling of the resource-intensive Visual Director
     * specialist service container on Cloud Run.
     */
    """
    from pitch_generator.app_utils.a2a import (
        A2aAgentExecutor,
        A2aAgentExecutorConfig,
        DefaultRequestHandler,
        include_artifacts_in_a2a_event_interceptor,
    )
    from pitch_generator.fast_api_app import PitchFastAPIApp

    resolved_services = services or get_default_services()
    app_instance = PitchFastAPIApp(services=resolved_services)

    card = build_visual_director_card(rpc_url=rpc_url)
    executor = A2aAgentExecutor(
        config=A2aAgentExecutorConfig(
            execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
        )
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store={},
        agent_card=card,
    )

    app_instance.register_a2a_handler(
        rpc_path="/a2a/visual_director",
        handler=handler,
        agent_card=card,
    )
    return app_instance


def _cloud_run_client(base_url: str) -> Any:
    """
    /**
     * Construct an HTTP client configured for Cloud Run A2A service communication.
     *
     * Why: Returns None for local http:// or empty URLs so mock/local networks work
     * without GCP credentials, and returns an authenticated client with 600s timeout
     * and OIDC bearer token for https:// Cloud Run service URLs.
     */
    """
    if not base_url or base_url.startswith("http://"):
        return None

    try:
        import google.auth
        import google.auth.transport.requests
        from google.oauth2 import id_token
        import httpx

        auth_req = google.auth.transport.requests.Request()
        token = id_token.fetch_id_token(auth_req, base_url)
        return httpx.Client(
            base_url=base_url,
            timeout=600.0,
            headers={"Authorization": f"Bearer {token}"},
        )
    except Exception:
        # Fall back to None in offline or non-GCP test environments
        return None


def _pitch_parts_only(part: Any) -> Any:
    """
    /**
     * Stream filter converter that strips tool-call chunks from outgoing A2A events.
     *
     * Why: Drops internal function_call and function_response parts (returning None/False)
     * while preserving clean text descriptions and rendered PNG key visual artifacts.
     */
    """
    if getattr(part, "function_call", None) is not None:
        return None
    if getattr(part, "function_response", None) is not None:
        return None
    return part


_vd_url: str = get_config().visual_director_url.rstrip("/")
remote_visual_director = RemoteA2aAgent(
    name="visual_director",
    description="Translates the campaign concept into art direction and key visuals via remote A2A microservice.",
    agent_card=f"{_vd_url}/.well-known/agent-card.json",
    httpx_client=_cloud_run_client(_vd_url),
    genai_part_converter=_pitch_parts_only,
    output_key="visual_director",
)

assemble = JoinNode(name="assemble")



def package(
    ctx_or_input: Any,
    node_input: dict[str, Any] | None = None,
) -> _DualSyncAsyncEventStream:
    """
    /**
     * Format and validate the joined specialist outputs into the final campaign pitch.
     *
     * Why: Acts as the terminal node after `assemble = JoinNode(name="assemble")`.
     * Explicitly checks that every expected upstream branch produced non-empty output
     * (raising `ValueError("nothing reached the join from: ...")` if any branch is
     * missing or empty) so silent branch failures are caught immediately. Supports both
     * `package(node_input)` and `package(ctx, node_input)` signatures, strips Markdown
     * code fences from branch values, and attaches `_key_visual(ctx)` when present.
     *
     * @param ctx_or_input Either a `node_input` dict (1-arg form) or `Context` (2-arg form).
     * @param node_input Dictionary of upstream outputs when `ctx_or_input` is a `Context`.
     * @return Dual sync/async stream yielding `Event(content=...)` and `Event(output=...)`.
     */
    """
    if node_input is None:
        if not isinstance(ctx_or_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx: Context | None = None
        actual_input = ctx_or_input
    else:
        if not isinstance(node_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx = ctx_or_input if isinstance(ctx_or_input, Context) else None
        actual_input = node_input

    if not actual_input:
        raise ValueError("nothing reached the join from: creative_director, copywriter")

    required_keys = ["creative_director", "copywriter"]
    missing_or_empty = [
        k
        for k in required_keys
        if k not in actual_input or not str(actual_input.get(k) or "").strip()
    ]
    extra_empty = [
        name
        for name, value in actual_input.items()
        if name not in missing_or_empty and not str(value or "").strip()
    ]
    all_empty = missing_or_empty + extra_empty
    if all_empty:
        raise ValueError(f"nothing reached the join from: {', '.join(all_empty)}")

    concept_val = strip_markdown_fences(str(actual_input["creative_director"])).strip()
    copy_val = strip_markdown_fences(str(actual_input["copywriter"])).strip()
    sections = [
        f"CONCEPT\n{concept_val}",
        f"COPY\n{copy_val}",
    ]
    if "brand_strategist" in actual_input and actual_input["brand_strategist"]:
        brand_val = strip_markdown_fences(str(actual_input["brand_strategist"])).strip()
        sections.append(f"BRAND STRATEGY\n{brand_val}")
    if "visual_director" in actual_input and actual_input["visual_director"]:
        art_val = strip_markdown_fences(str(actual_input["visual_director"])).strip()
        sections.append(f"ART DIRECTION\n{art_val}")

    pitch = "\n\n".join(sections)
    parts: list[Any] = [types.Part(text=pitch)]
    image = _key_visual(ctx)
    if image is not None:
        parts.append(types.Part(inline_data=image))

    events = [
        Event(author="package", content=types.Content(role="model", parts=parts)),
        Event(author="package", output=pitch),
    ]
    return _DualSyncAsyncEventStream(events)


loop_guard = LoopGuard(max_iterations=10)

# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]
# TODO (Step 3a): Implement a fail-closed `PreToolUse` security hook before tool execution:
#   - `DEFAULT_TOOL_ALLOWLIST = frozenset({"load_skill", "generate_key_visual"})`
#   - `ToolAuthorizationError(PermissionError)` with `.tool_name` and `.reason` attributes.
#   - `HookDecision` dataclass with `allowed: bool`, `tool_name: str`, `reason: str`,
#     `violation_type: str | None = None`, `sanitized_args: dict[str, Any]`, plus `.is_allowed`,
#     `.permitted`, and `__bool__`.
#   - `PreToolUseHook(allowlist=None, raise_on_block=False)` and module-level
#     `validate_tool_call(tool_name, tool_args=None) -> HookDecision`:
#     1. Block empty or unlisted tools (`violation_type="unauthorized_tool"`).
#     2. Block forbidden override keys such as `bypass_safety` or `override_brand_guidelines`
#        (`violation_type="forbidden_parameter"`).
#     3. Block path traversal sequences (`..` or leading `/`, `violation_type="path_traversal"`).
#     4. Block oversized string arguments (`> 4096` chars) and prompt injection patterns such as
#        `"ignore previous instructions"` or `system` tags (`violation_type="prompt_injection"`).
#     5. Append every check to `self.audit_log` and raise `ToolAuthorizationError` when `raise_on_block=True`.

DEFAULT_TOOL_ALLOWLIST: frozenset[str] = frozenset({"load_skill", "generate_key_visual"})


class ToolAuthorizationError(PermissionError):
    """
    /**
     * Exception raised when a tool invocation is blocked by the PreToolUse security hook.
     *
     * Why: Subclasses PermissionError to signal an unauthorized tool or argument violation
     * in a fail-closed manner.
     */
    """

    def __init__(self, message: str, tool_name: str = "", reason: str = ""):
        super().__init__(message)
        self.tool_name = tool_name
        self.reason = reason or message


@dataclass
class HookDecision:
    """
    /**
     * Represents the outcome of a PreToolUse lifecycle inspection.
     *
     * Why: Encapsulates authorization status, audit metadata, sanitized parameters,
     * and boolean interoperability for both programmatic checks and logging.
     */
    """

    allowed: bool
    tool_name: str
    reason: str
    violation_type: str | None = None
    sanitized_args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.allowed

    @property
    def permitted(self) -> bool:
        return self.allowed

    def __bool__(self) -> bool:
        return self.allowed


class PreToolUseHook:
    """
    /**
     * Fail-closed lifecycle hook intercepting tool calls prior to execution.
     *
     * Why: Enforces tool allowlists and argument sanitization to prevent unauthorized
     * tool execution, prompt injection, and path traversal attacks.
     */
    """

    FORBIDDEN_OVERRIDE_KEYS: frozenset[str] = frozenset(
        {
            "bypass_safety",
            "override_brand_guidelines",
            "disable_guardrails",
            "skip_validation",
            "admin",
            "root",
        }
    )

    INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?prior\s+instructions", re.IGNORECASE),
        re.compile(r"<\s*system\s*>", re.IGNORECASE),
        re.compile(r"<\s*/\s*system\s*>", re.IGNORECASE),
        re.compile(r"reveal\s+.*system\s+prompt", re.IGNORECASE),
    )

    def __init__(
        self,
        allowlist: frozenset[str] | Sequence[str] | set[str] | None = None,
        raise_on_block: bool = False,
    ) -> None:
        self.allowlist = frozenset(allowlist) if allowlist is not None else DEFAULT_TOOL_ALLOWLIST
        self.raise_on_block = raise_on_block
        self.audit_log: list[dict[str, Any]] = []

    def validate(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any] | None = None,
    ) -> HookDecision:
        cleaned_tool = (tool_name or "").strip()
        args = copy.deepcopy(tool_args) if isinstance(tool_args, dict) else {}

        # 1. Tool allowlist check
        if not cleaned_tool or cleaned_tool not in self.allowlist:
            decision = HookDecision(
                allowed=False,
                tool_name=cleaned_tool,
                reason=f"Tool {cleaned_tool!r} is not in allowed tools: {sorted(self.allowlist)}",
                violation_type="unauthorized_tool",
                sanitized_args={},
            )
            return self._record_and_handle(decision)

        # 2. Inspect tool arguments
        for k, v in args.items():
            k_clean = str(k).strip().lower()
            if k_clean in self.FORBIDDEN_OVERRIDE_KEYS or "bypass" in k_clean:
                decision = HookDecision(
                    allowed=False,
                    tool_name=cleaned_tool,
                    reason=f"Forbidden override parameter detected: {k!r}",
                    violation_type="forbidden_parameter",
                    sanitized_args={},
                )
                return self._record_and_handle(decision)

            if isinstance(v, str):
                # 3. Path traversal check
                if ".." in v or v.startswith("/") or v.startswith("\\"):
                    decision = HookDecision(
                        allowed=False,
                        tool_name=cleaned_tool,
                        reason=f"Path traversal sequence detected in parameter {k!r}: {v!r}",
                        violation_type="path_traversal",
                        sanitized_args={},
                    )
                    return self._record_and_handle(decision)

                # 4. Oversized argument check
                if len(v) > 4096:
                    decision = HookDecision(
                        allowed=False,
                        tool_name=cleaned_tool,
                        reason=f"Argument {k!r} exceeds maximum permitted length of 4096 characters",
                        violation_type="prompt_injection",
                        sanitized_args={},
                    )
                    return self._record_and_handle(decision)

                # Prompt injection patterns
                for pat in self.INJECTION_PATTERNS:
                    if pat.search(v):
                        decision = HookDecision(
                            allowed=False,
                            tool_name=cleaned_tool,
                            reason=f"Prompt injection pattern detected in {k!r}: {v!r}",
                            violation_type="prompt_injection",
                            sanitized_args={},
                        )
                        return self._record_and_handle(decision)

        # If all checks pass
        decision = HookDecision(
            allowed=True,
            tool_name=cleaned_tool,
            reason="Tool call validated against allowlist and argument constraints.",
            violation_type=None,
            sanitized_args=args,
        )
        return self._record_and_handle(decision)

    def _record_and_handle(self, decision: HookDecision) -> HookDecision:
        audit_entry = {
            "tool_name": decision.tool_name,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "violation_type": decision.violation_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.audit_log.append(audit_entry)

        if not decision.allowed and self.raise_on_block:
            raise ToolAuthorizationError(
                message=decision.reason,
                tool_name=decision.tool_name,
                reason=decision.reason,
            )
        return decision


_GLOBAL_PRE_TOOL_USE_HOOK = PreToolUseHook()


def validate_tool_call(
    tool_name: str | None,
    tool_args: dict[str, Any] | None = None,
    hook: PreToolUseHook | None = None,
) -> HookDecision:
    """
    /**
     * Module-level helper validating a tool call against the PreToolUse security hook.
     *
     * Why: Provides a direct function entrypoint for agent tool invocation guards.
     */
    """
    active_hook = hook or _GLOBAL_PRE_TOOL_USE_HOOK
    return active_hook.validate(tool_name, tool_args)


# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]
# TODO (Step 3c): Add the two-node ADK Human-in-the-Loop checkpoint after `creative_director`:
#   - `@node(rerun_on_resume=False) async def approve_concept(ctx: Context)`:
#     validates that `ctx.session.state.get("creative_director")` is non-empty and yields
#     `RequestInput(message="Please approve the campaign concept (yes/no).", response_schema=str)`.
#   - `evaluate_user_approval(concept: str, decision: str) -> str`:
#     accepts affirmative responses (`"yes"`, `"y"`, `"true"`, `"approved"` -> returns
#     `f"## Approved Concept\n\n{concept}"`) and raises `ValueError("User rejected the concept")` otherwise.
#   - `@node(rerun_on_resume=True) async def user_approval(ctx: Context, node_input: dict[str, Any] | None = None)`:
#     awaits `ctx.run_node(approve_concept)`, validates the decision via `evaluate_user_approval`,
#     and yields `Event(author="user_approval", content=...)`.
#   - `run_hitl_workflow(brief: str, *, approved: bool | None = None, services: ServiceContainer | None = None) -> dict[str, Any]`:
#     returns `status="input_required"` (`downstream_calls=0`) when `approved is None`,
#     `status="rejected"` (`downstream_calls=0`) when `approved is False`, and `status="completed"`
#     (`downstream_calls=2`, `"## Approved Concept"` in `concept_header`) when `approved is True`.
#   - Insert `user_approval` into `root_agent.edges` after `creative_director`, and enable the
#     Human-in-the-Loop UI controls (`#hitl-checkbox-group` and `#hitl-approval-card`) in
#     `frontend/index.html` by removing their `hidden` attributes (defining `approve_concept` or
#     `user_approval` here also automatically sets `hitl_enabled=True` in `/api/config` and `/api/pitch`).


def evaluate_user_approval(concept: str, decision: str) -> str:
    """
    /**
     * Validate human approval decision for a campaign concept.
     *
     * Why: Accepts affirmative responses ('yes', 'y', 'true', 'approved'), returning
     * formatted markdown header with '## Approved Concept' or raising ValueError on rejection.
     *
     * @param concept The generated creative concept string.
     * @param decision The raw user decision input string.
     * @return Formatted concept string prefixed with '## Approved Concept'.
     */
    """
    clean_decision = str(decision or "").strip().lower()
    if clean_decision in {"yes", "y", "true", "approved", "1"}:
        return f"## Approved Concept\n\n{concept}"
    raise ValueError("User rejected the concept")


@node(rerun_on_resume=False)
async def approve_concept(ctx: Context) -> Any:
    """
    /**
     * Human-in-the-loop checkpoint pausing execution after creative_director ideation.
     *
     * Why: Yields RequestInput so the orchestrator halts execution until an external
     * human review event arrives.
     */
    """
    concept = ctx.session.state.get("creative_director")
    if not concept:
        raise ValueError("Cannot request approval: creative_director state is empty.")
    yield RequestInput(
        message="Please approve the campaign concept (yes/no).",
        response_schema=str,
    )


@node(rerun_on_resume=True)
async def user_approval(ctx: Context, node_input: dict[str, Any] | None = None) -> Any:
    """
    /**
     * Evaluation node resuming on human input and gating downstream generation branches.
     *
     * Why: Awaits the human decision from `approve_concept`, validates it via
     * `evaluate_user_approval`, and emits the authorized concept event or aborts execution.
     */
    """
    concept = ctx.session.state.get("creative_director", "")
    decision_input = await ctx.run_node(approve_concept)
    decision_str = str(decision_input or "").strip()
    approved_concept = evaluate_user_approval(concept, decision_str)
    yield Event(author="user_approval", content=approved_concept)


def run_hitl_workflow(
    brief: str,
    *,
    approved: bool | None = None,
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Standalone runner simulating the Human-in-the-Loop approval lifecycle.
     *
     * Why: Demonstrates the three HITL lifecycle states:
     * - `approved is None`: pauses after ideation with `status="input_required"`, 0 downstream calls.
     * - `approved is False`: halts with `status="rejected"`, 0 downstream calls.
     * - `approved is True`: continues with `status="completed"`, 2 downstream calls (`copywriter`, `visual_director`).
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must not be empty")

    svc = services or get_default_services()
    model_id = svc.config.flash_model

    # Phase 1: Ideation (Creative Director)
    raw_concept = svc.llm.generate_text(
        brief.strip(),
        system_instruction=creative_director.instruction,
        model=model_id,
    )
    concept = strip_markdown_fences(raw_concept).strip()

    if approved is None:
        return {
            "status": "input_required",
            "prompt": "Please approve the campaign concept (yes/no).",
            "concept": concept,
            "downstream_calls": 0,
        }

    if approved is False:
        return {
            "status": "rejected",
            "concept": concept,
            "downstream_calls": 0,
        }

    # approved is True
    approved_header = evaluate_user_approval(concept, "yes")
    return {
        "status": "completed",
        "concept": approved_header,
        "concept_header": approved_header,
        "downstream_calls": 2,
    }


root_agent = Workflow(
    name="pitch_generator",
    edges=[
        (creative_director, user_approval, brand_strategist, (copywriter, remote_visual_director)),
        ((copywriter, remote_visual_director), assemble, package),
    ],
)
loop_guard.validate_graph(root_agent.edges)

app = App(root_agent=root_agent, name="pitch_generator")


# [Guidepost — Step 4a: Tokenomics & Context Optimization]
# TODO (Step 4a): Implement memory compression, sliding-window pruning, SHA-256 prompt caching, and model routing:
#   - `estimate_tokens(text: str | None) -> int`: returns `0` for empty/whitespace text, else `max(1, len(text.strip()) // 4)`.
#   - `CompressedHistoryList(list)`: `list` subclass with `.turns`, `.history`, `.summary`,
#     `.original_tokens`, `.optimized_tokens`, and `.tokens_saved` attributes.
#   - `compress_memory(turns: Sequence[dict[str, Any]], keep_recent: int = 2) -> CompressedHistoryList`:
#     summarizes older turns prior to the last `keep_recent` turns into a single
#     `{"role": "system", "compressed": True, "content": ...}` entry while preserving the most recent turns verbatim.
#   - `prune_history(turns: Sequence[dict[str, Any]], max_turns: int = 6, max_tokens: int = 512) -> CompressedHistoryList`:
#     validates `max_turns > 0` and `max_tokens > 0` (raising `ValueError` otherwise), keeps the newest turns
#     within the turn/token budget, and truncates a single oversized turn (`"truncated": True`) to fit `max_tokens`.
#   - `PromptCacheManager(default_ttl_seconds: int = 300)`: `.get_or_create(static_prompt: str, ttl_seconds: int | None = None)`
#     validates non-empty text, computes a 64-char SHA-256 hex `cache_key` (`hashlib.sha256(...).hexdigest()`),
#     and tracks TTL expiration, `"cache_hit"` (`True`/`False`), and `"saved_tokens"`.
#   - `select_model_strategy(tier: str = "flash") -> dict[str, Any]` supporting tiers
#     `("flash", "pro", "low", "high", "local")` (raising `ValueError` on unknown tiers) and returning
#     `"tier"`, `"model_id"`, `"target"`, `"cost_tier"`, and `"thinking_budget"`.
#   - `TokenomicsManager(max_turns: int = 6, max_tokens: int = 512)`: `.optimize(turns, static_prompt="", max_turns=..., max_tokens=..., tier="flash")`
#     combining `compress_memory`, `prune_history`, `PromptCacheManager`, and `select_model_strategy`.


def estimate_tokens(text: str | None) -> int:
    """
    /**
     * Heuristic token estimator assuming ~4 characters per token for English text.
     *
     * Why: Provides a fast, deterministic, offline-capable token counter without
     * requiring heavy external tokenizer dependencies.
     *
     * @param text Input text string or None.
     * @return Estimated integer token count (0 for empty/whitespace).
     */
    """
    if not text or not str(text).strip():
        return 0
    return max(1, len(str(text).strip()) // 4)


class CompressedHistoryList(list):
    """
    /**
     * List subclass storing conversation turns along with token optimization metadata.
     *
     * Why: Preserves standard list semantics for agent runners while tracking
     * original tokens, optimized tokens, saved tokens, and summary state.
     */
    """

    def __init__(
        self,
        iterable: Sequence[dict[str, Any]] | None = None,
        *,
        summary: str = "",
        original_tokens: int = 0,
        optimized_tokens: int = 0,
    ) -> None:
        super().__init__(iterable or [])
        self.summary = summary
        self.original_tokens = original_tokens
        self.optimized_tokens = (
            optimized_tokens
            if optimized_tokens > 0 or not self
            else sum(estimate_tokens(t.get("content", "")) for t in self)
        )
        self.tokens_saved = max(0, self.original_tokens - self.optimized_tokens)

    @property
    def turns(self) -> list[dict[str, Any]]:
        return list(self)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self)


def compress_memory(
    turns: Sequence[dict[str, Any]],
    keep_recent: int = 2,
) -> CompressedHistoryList:
    """
    /**
     * Compress earlier dialogue turns into a single synthetic system summary turn.
     *
     * Why: Keeps the prompt context bounded while preserving salient facts from
     * older interactions without overflowing token limits.
     *
     * @param turns Sequence of conversation turn dictionaries.
     * @param keep_recent Number of most recent turns to preserve verbatim.
     * @return CompressedHistoryList with summarized prefix and recent turns verbatim.
     */
    """
    turns_list = list(turns or [])
    if not turns_list:
        return CompressedHistoryList([], summary="", original_tokens=0, optimized_tokens=0)

    original_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns_list)

    if len(turns_list) <= keep_recent or keep_recent <= 0:
        return CompressedHistoryList(
            [copy.deepcopy(t) for t in turns_list],
            summary="",
            original_tokens=original_tokens,
            optimized_tokens=original_tokens,
        )

    split_idx = len(turns_list) - keep_recent
    older_turns = turns_list[:split_idx]
    recent_turns = turns_list[split_idx:]

    summary_parts = []
    for turn in older_turns:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        summary_parts.append(f"{role}: {content}")
    summary_text = " | ".join(summary_parts)
    compressed_content = f"Summary of earlier conversation: {summary_text}"

    summary_turn = {
        "role": "system",
        "compressed": True,
        "content": compressed_content,
    }

    result_turns = [summary_turn] + [copy.deepcopy(t) for t in recent_turns]
    optimized_tokens = sum(estimate_tokens(t.get("content", "")) for t in result_turns)

    return CompressedHistoryList(
        result_turns,
        summary=compressed_content,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
    )


def prune_history(
    turns: Sequence[dict[str, Any]],
    max_turns: int = 6,
    max_tokens: int = 512,
) -> CompressedHistoryList:
    """
    /**
     * Sliding-window history pruning enforcing hard limits on turn count and token budget.
     *
     * Why: Prevents runaway prompt expansion by retaining only the newest turns and
     * truncating oversized single turns with 'truncated': True.
     *
     * @param turns Input sequence of conversation turns.
     * @param max_turns Maximum allowed number of turns to keep.
     * @param max_tokens Maximum token budget for the kept history.
     * @return CompressedHistoryList containing pruned and budget-fitted turns.
     */
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    turns_list = list(turns or [])
    if not turns_list:
        return CompressedHistoryList([], summary="", original_tokens=0, optimized_tokens=0)

    original_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns_list)

    # Step 1: Keep at most max_turns from the newest end
    candidate_turns = turns_list[-max_turns:]

    # Step 2: Fit within max_tokens from newest backwards
    accumulated_turns: list[dict[str, Any]] = []
    tokens_used = 0

    for turn in reversed(candidate_turns):
        t_copy = copy.deepcopy(turn)
        content = str(t_copy.get("content", ""))
        turn_tokens = estimate_tokens(content)

        if tokens_used + turn_tokens <= max_tokens:
            accumulated_turns.append(t_copy)
            tokens_used += turn_tokens
        else:
            # Check if we can fit a truncated version into the remaining budget
            remaining_tokens = max_tokens - tokens_used
            if remaining_tokens > 0 and not accumulated_turns:
                # If even a single turn doesn't fit, truncate it to fit max_tokens
                max_chars = remaining_tokens * 4
                truncated_content = content[:max_chars]
                t_copy["content"] = truncated_content
                t_copy["truncated"] = True
                accumulated_turns.append(t_copy)
                tokens_used += estimate_tokens(truncated_content)
            break

    accumulated_turns.reverse()
    optimized_tokens = sum(estimate_tokens(t.get("content", "")) for t in accumulated_turns)

    return CompressedHistoryList(
        accumulated_turns,
        summary="",
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
    )


class PromptCacheManager:
    """
    /**
     * In-memory SHA-256 prompt cache manager tracking cache hits and token savings.
     *
     * Why: Mirrors Gemini Context Caching semantics by generating deterministic 64-char
     * hex sha256 digests of static prompt prefixes, enforcing TTL expiration, and
     * avoiding redundant token processing on repetitive calls.
     */
    """

    def __init__(self, default_ttl_seconds: int = 300) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}

    def get_or_create(
        self,
        static_prompt: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        /**
         * Lookup or register a static prompt in the cache using its SHA-256 digest.
         *
         * @param static_prompt Non-empty static prompt text to cache.
         * @param ttl_seconds Cache lifetime in seconds (defaults to default_ttl_seconds).
         * @return Dictionary with cache_key, cache_hit, saved_tokens, and metadata.
         */
        """
        if not static_prompt or not str(static_prompt).strip():
            raise ValueError("static_prompt must be a non-empty string.")

        clean_text = str(static_prompt).strip()
        cache_key = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        tokens = estimate_tokens(clean_text)

        entry = self._cache.get(cache_key)
        if entry is not None and entry["expires_at"] > now:
            # Cache Hit
            return {
                "cache_key": cache_key,
                "cache_hit": True,
                "saved_tokens": tokens,
                "expires_at": entry["expires_at"],
                "tokens": tokens,
            }

        # Cache Miss
        expires_at = now + ttl
        self._cache[cache_key] = {
            "prompt": clean_text,
            "tokens": tokens,
            "created_at": now,
            "expires_at": expires_at,
        }
        return {
            "cache_key": cache_key,
            "cache_hit": False,
            "saved_tokens": 0,
            "expires_at": expires_at,
            "tokens": tokens,
        }


def select_model_strategy(tier: str = "flash") -> dict[str, Any]:
    """
    /**
     * Resolve cost and latency model strategy configuration by tier.
     *
     * Why: Routes simple low-latency tasks to flash / low models and high-reasoning
     * multi-turn synthesis to pro / high models while specifying thinking budgets.
     *
     * @param tier Model tier string ("flash", "pro", "low", "high", "local").
     * @return Dictionary with tier, model_id, target, cost_tier, and thinking_budget.
     */
    """
    tier_clean = str(tier or "flash").strip().lower()
    strategy_map: dict[str, dict[str, Any]] = {
        "flash": {
            "tier": "flash",
            "model_id": "gemini-3.8-flash",
            "target": "cloud",
            "cost_tier": "low",
            "thinking_budget": 0,
        },
        "low": {
            "tier": "low",
            "model_id": "gemini-3.8-flash",
            "target": "cloud",
            "cost_tier": "low",
            "thinking_budget": 0,
        },
        "pro": {
            "tier": "pro",
            "model_id": "gemini-3.8-pro",
            "target": "cloud",
            "cost_tier": "high",
            "thinking_budget": 1024,
        },
        "high": {
            "tier": "high",
            "model_id": "gemini-3.8-pro",
            "target": "cloud",
            "cost_tier": "high",
            "thinking_budget": 1024,
        },
        "local": {
            "tier": "local",
            "model_id": "gemini-nano-banana-2.1",
            "target": "local",
            "cost_tier": "zero",
            "thinking_budget": 0,
        },
    }

    if tier_clean not in strategy_map:
        raise ValueError(
            f"Unknown model tier: {tier!r}. Supported tiers: {tuple(strategy_map.keys())}"
        )

    return copy.deepcopy(strategy_map[tier_clean])


class TokenomicsManager:
    """
    /**
     * Unified manager orchestrating context optimization, caching, and model routing.
     *
     * Why: Combines memory compression, sliding-window history pruning, prompt caching,
     * and tier-based model routing into a single optimize call for agent execution pipelines.
     */
    """

    def __init__(self, max_turns: int = 6, max_tokens: int = 512) -> None:
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.cache_manager = PromptCacheManager()

    def optimize(
        self,
        turns: Sequence[dict[str, Any]],
        *,
        static_prompt: str = "",
        max_turns: int | None = None,
        max_tokens: int | None = None,
        tier: str = "flash",
    ) -> dict[str, Any]:
        """
        /**
         * Optimize prompt turns and static instructions for agent execution.
         *
         * @param turns Input sequence of conversation turns.
         * @param static_prompt Optional system instructions or static prefix.
         * @param max_turns Optional turn count override.
         * @param max_tokens Optional token budget override.
         * @param tier Model tier selection ("flash", "pro", etc.).
         * @return Optimization summary dictionary with pruned turns, cache info, and model strategy.
         */
        """
        effective_max_turns = max_turns if max_turns is not None else self.max_turns
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # 1. Compress memory
        compressed = compress_memory(turns, keep_recent=2)

        # 2. Prune history
        pruned = prune_history(
            compressed,
            max_turns=effective_max_turns,
            max_tokens=effective_max_tokens,
        )

        # 3. Prompt caching
        cache_info = (
            self.cache_manager.get_or_create(static_prompt)
            if static_prompt and str(static_prompt).strip()
            else {"cache_hit": False, "saved_tokens": 0, "cache_key": ""}
        )

        # 4. Model strategy
        strategy = select_model_strategy(tier)

        total_saved = pruned.tokens_saved + cache_info.get("saved_tokens", 0)

        return {
            "turns": pruned.turns,
            "summary": compressed.summary,
            "original_tokens": pruned.original_tokens,
            "optimized_tokens": pruned.optimized_tokens,
            "tokens_saved": total_saved,
            "cache": cache_info,
            "strategy": strategy,
        }


# ==============================================================================
# Internal Workflow Execution Engine (Do not modify unless extending runners)
# ==============================================================================


def run_graph_workflow(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    max_iterations: int = 10,
) -> PitchPackage:
    """
    /**
     * Validates the workflow DAG with `LoopGuard` and executes the fan-out/fan-in pitch graph.
     *
     * Why: Provides a deterministic, offline-safe execution helper that validates the
     * graph topology, tracks every node visit in `LoopGuard`, parses structured payloads,
     * and returns a complete `PitchPackage`.
     *
     * @param brief Campaign product brief string.
     * @param services Optional `ServiceContainer` for offline LLM/artifact services.
     * @param max_iterations Maximum allowed node visits in `LoopGuard`.
     * @return Populated `PitchPackage` instance.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string.")

    guard = LoopGuard(max_iterations=max_iterations)
    guard.validate_graph(root_agent.edges)

    svc = services or get_default_services()
    ctx = Context(services=svc)
    clean_brief = brief.strip()
    model_id = svc.config.flash_model

    guard.record_step("creative_director")
    concept_raw = svc.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=model_id,
    )
    concept_lines = [ln.strip() for ln in concept_raw.splitlines() if ln.strip()]
    concept = ConceptPayload(
        concept_line=concept_lines[0] if concept_lines else f"Engineered for everyday motion: {clean_brief}.",
        rationale=(
            concept_lines[1]
            if len(concept_lines) > 1
            else "Grounded in practical commuter utility and understated design."
        ),
    )

    guard.record_step("brand_strategist")
    strategy_notes = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=brand_strategist.instruction,
        model=model_id,
    )

    guard.record_step("copywriter")
    copy_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=copywriter.instruction,
        model=model_id,
    )
    copy_payload = CopyPayload(caption=copy_text)

    guard.record_step("visual_director")
    art_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=visual_director.instruction,
        model=model_id,
    )
    art_payload = ArtDirectionPayload(art_direction=art_text)

    guard.record_step("assemble")
    joined_inputs = {
        "creative_director": f"{concept.concept_line}\n{concept.rationale}",
        "brand_strategist": strategy_notes,
        "copywriter": copy_payload.caption,
        "visual_director": art_payload.art_direction,
    }

    guard.record_step("package")
    packaged_events = list(package(ctx, joined_inputs))
    final_text = ""
    for ev in packaged_events:
        if ev.output:
            final_text = str(ev.output)

    return PitchPackage(
        concept=concept,
        copy=copy_payload,
        art_direction=art_payload,
        brand_strategy=strategy_notes,
        pitch_text=final_text,
        trace=list(guard.visited_nodes),
    )


execute_workflow = run_graph_workflow
run_workflow = run_graph_workflow


def run_pitch_workflow(
    brief: str,
    *,
    session_id: str = "default",
    services: ServiceContainer | None = None,
    approved: bool | None = True,
    require_approval: bool = False,
    existing_concept: str | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the Pitch Generator multi-agent workflow using injected cloud services.
     *
     * Why: Coordinates the 4-specialist pipeline (`creative_director` -> `brand_strategist`
     * -> (`copywriter`, `visual_director`) -> `assemble` -> `package`) on Gemini Enterprise
     * Agent Platform (`gemini-3.8-flash`), persists session state in `MemoryBankService`,
     * and records telemetry via `BigQueryAnalyticsService`. When the learner adds the HITL
     * approval gate (`approve_concept` / `user_approval`) in Module 3, this runner dynamically
     * pauses and resumes at the concept review gate.
     *
     * @param brief Non-empty campaign topic brief.
     * @param session_id Session identifier for Memory Bank persistence.
     * @param services Optional injected `ServiceContainer` (defaults to `get_default_services()`).
     * @param approved HITL approval state (`True` to complete, `False` to reject,
     *   `None` to pause at `input_required` when HITL is enabled).
     * @param require_approval If True, pauses at the HITL gate (`status="input_required"`).
     * @param existing_concept Optional pre-generated concept to reuse when resuming a paused
     *   HITL session so `creative_director` is not re-executed.
     * @return Workflow result dictionary with keys `session_id`, `status`, `concept`,
     *   `copy`, `brand_strategy`, `art_direction`, `key_visual_uri`, `telemetry`,
     *   `artifacts`, `pitch_text`, and `trace`.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must not be empty")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")

    clean_brief = brief.strip()
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    module_globals = globals()
    strip_fn = module_globals.get("strip_markdown_fences")
    active_model = active_services.config.flash_model

    # Step 1: Creative Director generates campaign concept & rationale (or reuses saved concept on resume)
    if isinstance(existing_concept, str) and existing_concept.strip():
        concept = existing_concept.strip()
    else:
        concept = active_services.llm.generate_text(
            clean_brief,
            system_instruction=creative_director.instruction,
            model=active_model,
        )
        if callable(strip_fn):
            concept = strip_fn(concept)

    # Determine effective approval state (used when learner enables HITL in Module 3)
    effective_approved: bool | None = approved
    if require_approval and approved is True:
        effective_approved = None

    if effective_approved is False:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "rejected",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept", "user_approval"],
            }
        )
        rejected_state = {
            "session_id": clean_session,
            "status": "rejected",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept", "user_approval"],
        }
        active_services.memory_bank.save_session(clean_session, rejected_state)
        return rejected_state

    if effective_approved is None:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "input_required",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept"],
            }
        )
        paused_state = {
            "session_id": clean_session,
            "status": "input_required",
            "prompt": "Please approve the campaign concept (yes/no).",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept"],
        }
        active_services.memory_bank.save_session(clean_session, paused_state)
        return paused_state

    # Step 2: Copywriter generates <25-word social caption
    copy_text = active_services.llm.generate_text(
        concept,
        system_instruction=copywriter.instruction,
        model=active_model,
    )

    hitl_defined = (
        module_globals.get("approve_concept") is not None
        or module_globals.get("user_approval") is not None
    )
    join_inputs: dict[str, Any] = {
        "creative_director": concept,
        "copywriter": copy_text,
    }
    nodes_executed: list[str] = (
        ["creative_director", "approve_concept", "user_approval", "copywriter"]
        if hitl_defined
        else ["creative_director", "copywriter"]
    )
    brand_text = ""
    art_direction = ""
    key_visual_uri: str | None = None
    key_visual_url: str | None = None
    artifacts_meta: list[dict[str, Any]] = []

    loop_guard_cls = module_globals.get("LoopGuard")
    active_guard = None
    if callable(loop_guard_cls):
        active_guard = loop_guard_cls(max_iterations=10)
        if hasattr(active_guard, "validate_graph") and hasattr(root_agent, "edges"):
            active_guard.validate_graph(root_agent.edges)
        if hasattr(active_guard, "record_step"):
            active_guard.record_step("creative_director")
            active_guard.record_step("copywriter")

    if callable(strip_fn):
        copy_text = strip_fn(copy_text)
        join_inputs["creative_director"] = concept
        join_inputs["copywriter"] = copy_text

    brand_agent = module_globals.get("brand_strategist")
    if brand_agent is None:
        brand_agent = next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "brand_strategist"
            ),
            None,
        )
    if brand_agent is not None:
        brand_instruction = getattr(
            brand_agent,
            "instruction",
            "You are the Brand Strategist. Provide audience positioning for the concept.",
        )
        brand_text = active_services.llm.generate_text(
            concept,
            system_instruction=brand_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            brand_text = strip_fn(brand_text)
        join_inputs["brand_strategist"] = brand_text
        nodes_executed.append("brand_strategist")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("brand_strategist")

    remote_vd = module_globals.get("remote_visual_director")
    if remote_vd is None:
        remote_vd = next(
            (v for v in module_globals.values() if isinstance(v, RemoteA2aAgent)),
            None,
        )
    visual_agent = (
        module_globals.get("visual_director")
        or next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "visual_director"
            ),
            None,
        )
        or remote_vd
    )
    if visual_agent is not None:
        if remote_vd is not None:
            from pitch_generator.app_utils.services import _is_offline_test_mode
            import urllib.request

            if not _is_offline_test_mode(active_services.config.project_id):
                vd_url = active_services.config.visual_director_url.rstrip("/")
                card_url = f"{vd_url}/.well-known/agent-card.json"
                headers: dict[str, str] = {"Accept": "application/json"}
                if card_url.startswith("https://"):
                    try:
                        import google.auth.transport.requests
                        import google.oauth2.id_token

                        auth_req = google.auth.transport.requests.Request()
                        id_tok = google.oauth2.id_token.fetch_id_token(auth_req, vd_url)
                        if id_tok:
                            headers["Authorization"] = f"Bearer {id_tok}"
                    except Exception:
                        pass
                try:
                    req = urllib.request.Request(card_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}")
                except Exception as exc:
                    raise RuntimeError(
                        f"[A2A Remote Error] Could not connect to remote Visual Director service at "
                        f"'{vd_url}' ({card_url}): {exc}. Ensure the Visual Director service is running "
                        f"(or verify VISUAL_DIRECTOR_URL in .env)."
                    ) from exc

        visual_instruction = str(
            getattr(
                visual_agent,
                "instruction",
                (
                    "You are the Visual Director. Produce brand-compliant art direction "
                    "using our deep indigo and slate palette with a warm amber or terracotta "
                    "accent, single low raking light with long shadows, off-center negative "
                    "space, and one realistic photographic subject."
                ),
            )
            or ""
        )
        skill_md_path = Path(__file__).resolve().parent / "skills" / "brand-guidelines" / "SKILL.md"
        if skill_md_path.is_file():
            try:
                skill_text = skill_md_path.read_text(encoding="utf-8").strip()
                if skill_text and skill_text not in visual_instruction:
                    visual_instruction = f"{visual_instruction}\n\n[Loaded Skill: brand-guidelines]\n{skill_text}"
            except OSError:
                pass
        effective_visual_instruction = (
            f"{visual_instruction}\n\n"
            "Return ONLY the written art direction notes as plain text "
            "(do not emit tool call syntax; generate_key_visual will be invoked automatically with your text output)."
        )
        art_direction = active_services.llm.generate_text(
            concept,
            system_instruction=effective_visual_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            art_direction = strip_fn(art_direction)
        image_bytes, mime_type = active_services.llm.generate_image(
            art_direction,
            model=active_services.config.image_model,
        )
        artifact_record = active_services.artifacts.save_artifact(
            filename="key_visual.png",
            data=image_bytes,
            mime_type=mime_type,
            session_id=clean_session,
        )
        key_visual_url = (
            f"/api/artifacts/{clean_session}/{artifact_record.filename}"
            f"?v={artifact_record.version}"
        )
        key_visual_uri = (
            artifact_record.gcs_uri
            or f"/api/artifacts/{clean_session}/{artifact_record.filename}"
        )
        artifacts_meta.append(artifact_record.to_metadata_dict())
        join_inputs["visual_director"] = art_direction
        nodes_executed.append("visual_director")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("visual_director")

    # Step 3: Assemble and Package join output validation
    join_events = list(package(join_inputs))
    pitch_text = str(join_events[-1].output or "")
    nodes_executed.extend(["assemble", "package"])
    if active_guard is not None and hasattr(active_guard, "record_step"):
        active_guard.record_step("assemble")
        active_guard.record_step("package")

    telemetry_event = active_services.analytics.record_telemetry(
        {
            "session_id": clean_session,
            "status": "completed",
            "brief": clean_brief,
            "model_id": active_model,
            "key_visual_uri": key_visual_uri,
            "nodes_executed": list(nodes_executed),
        }
    )

    completed_result = {
        "session_id": clean_session,
        "status": "completed",
        "brief": clean_brief,
        "concept": concept,
        "copy": copy_text,
        "brand_strategy": brand_text,
        "art_direction": art_direction,
        "key_visual_uri": key_visual_uri,
        "key_visual_url": key_visual_url,
        "telemetry": telemetry_event,
        "artifacts": artifacts_meta,
        "pitch_text": pitch_text,
        "trace": list(nodes_executed),
    }

    active_services.memory_bank.save_session(clean_session, completed_result)
    active_services.memory_bank.store_memory(
        "campaigns",
        clean_session,
        {
            "brief": clean_brief,
            "concept": concept,
            "copy": copy_text,
            "key_visual_uri": key_visual_uri,
        },
    )
    return completed_result


def resume_pitch_workflow(
    session_id: str,
    *,
    approved: bool,
    feedback: str = "",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Resume a paused Human-in-the-Loop workflow session stored in `MemoryBankService`.
     *
     * Why: When `POST /api/approve` or an A2A approval message arrives for a paused
     * `session_id` (Module 3), this function loads the saved concept from Memory Bank
     * and either completes the downstream workflow (`approved=True`) or marks the
     * session `"rejected"` (`approved=False`).
     *
     * @param session_id Session identifier of the paused workflow.
     * @param approved Boolean decision from the human reviewer.
     * @param feedback Optional reviewer note or refinement instruction.
     * @param services Optional injected `ServiceContainer`.
     * @return Updated workflow result dictionary (`"completed"` or `"rejected"`).
     */
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    stored = active_services.memory_bank.load_session(clean_session)
    if not stored:
        raise KeyError(f"Unknown session_id: {clean_session}")

    brief = str(stored.get("brief") or "Campaign brief")
    saved_concept = str(stored.get("concept") or "").strip() or None

    if not approved:
        result = run_pitch_workflow(
            brief,
            session_id=clean_session,
            services=active_services,
            approved=False,
            existing_concept=saved_concept,
        )
        result["reviewer_feedback"] = feedback
        active_services.memory_bank.save_session(clean_session, result)
        return result

    effective_brief = f"{brief} ({feedback.strip()})" if feedback and feedback.strip() else brief
    result = run_pitch_workflow(
        effective_brief,
        session_id=clean_session,
        services=active_services,
        approved=True,
        require_approval=False,
        existing_concept=saved_concept,
    )
    result["reviewer_feedback"] = feedback
    active_services.memory_bank.save_session(clean_session, result)
    return result
