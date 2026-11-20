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
 * @file services.py
 * @description Dependency-injected cloud service abstractions for Gemini Enterprise
 *   Agent Platform, Cloud Storage artifacts, and BigQuery analytics.
 *
 * Why: Isolating external cloud SDK interactions behind runtime-checkable protocols
 * (`LLMClientProtocol`, `ArtifactServiceProtocol`) and providing deterministic offline
 * implementations (`DeterministicMockLLMClient`, `InMemoryArtifactService`,
 * `GcsArtifactService`, `BigQueryAnalyticsService`) allows 100% of unit, functional,
 * and E2E tests to run offline without live GCP credentials or network calls (`R4`).
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/app_utils/services.py`)
#   - Step 2a: `# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]`
#   - Step 2b: `# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]`
#   - Step 3b: `# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]`
# (Tip: This file is < 450 lines. Read the entire file in a single `view_file` call.)
# ==============================================================================

import copy
from datetime import datetime, timezone
import re
from typing import Any

from pitch_generator.app_utils.memory_bank import MemoryBankService
from pitch_generator.app_utils.runtime_clients import (
    MINIMAL_PNG_BYTES,
    ArtifactRecord,
    ArtifactServiceProtocol,
    DeterministicMockLLMClient,
    EnterpriseGenAILLMClient,
    GcsArtifactService,
    InMemoryArtifactService,
    LLMClientProtocol,
    MockLLMClient,
    _build_house_brand_png,
    _is_live_cloud_configured,
    _is_offline_test_mode,
    _validate_artifact_inputs,
    create_artifact_service,
    get_artifact_service,
    resolve_artifact_service,
    select_artifact_service,
)
from pitch_generator.config import (
    DEFAULT_FLASH_MODEL,
    PitchConfig,
    get_config,
    normalize_bucket_name,
)


def run_workflow_with_gcs_artifacts(
    brief: str,
    *,
    bucket_name: str = "local-dev-project-bwg",
    session_id: str = "default",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the end-to-end Pitch Generator workflow backed by `GcsArtifactService`.
     *
     * Why: Verifies that `generate_key_visual` and the orchestrator pipeline persist
     * `key_visual.png` to `gs://<bucket>/key-visuals/key_visual.png` and surface the
     * `gs://` URI in the packaged output.
     *
     * @param brief Product pitch brief text.
     * @param bucket_name Target GCS bucket name (default `'local-dev-project-bwg'`).
     * @param session_id Session identifier (default `'default'`).
     * @param services Optional pre-built `ServiceContainer`.
     * @return Workflow execution result dictionary including `key_visual_uri` and `gcs_uri`.
     */
    """
    from pitch_generator.agent import run_pitch_workflow

    cfg = get_config(env={"LOGS_BUCKET_NAME": bucket_name})
    svc_container = services or get_default_services(cfg)
    svc_container.artifacts = GcsArtifactService(bucket_name=bucket_name)
    svc_container.artifact_service = svc_container.artifacts
    result = run_pitch_workflow(brief, session_id=session_id, services=svc_container)
    result["gcs_uri"] = result.get("key_visual_uri")
    return result


class BigQueryAnalyticsService:
    """
    /**
     * Injectable BigQuery Agent Analytics and brand compliance audit service.
     *
     * Why: Provides telemetry event logging, `ObjectRef` (`OBJ.MAKE_REF` and
     * `OBJ.FETCH_METADATA`) SQL generation for `bwg.key_visuals`, `AI.SCORE` SQL
     * generation for `bwg.brand_review`, and deterministic offline brand compliance
     * grading against the house brand guidelines.
     */
    """

    POSITIVE_BRAND_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("palette_indigo_slate", ("indigo", "slate")),
        ("warm_accent", ("amber", "terracotta")),
        ("lighting_raking", ("raking", "long shadow", "golden hour", "studio key")),
        ("composition_negative_space", ("off-center", "negative space", "third")),
        ("subject_photographic", ("photographic", "shallow depth of field", "single", "hero")),
    )

    FORBIDDEN_BRAND_TERMS: tuple[str, ...] = (
        "neon",
        "fluorescent",
        "watermark",
        "logo",
        "signage",
        "collage",
        "split-screen",
        "chrome",
        "lens flare",
        "bokeh sparkle",
        "cyan",
        "magenta",
        "fisheye",
        "dutch tilt",
        "flat lay",
        "ring light",
        "3d render",
        "pixel art",
        "clay",
    )

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        client: Any = None,
    ) -> None:
        """
        /**
         * Initialize the BigQuery analytics service.
         *
         * Why: Stores project, region, dataset (`bwg`), and Cloud Resource connection
         * (`pitch-connection`) identifiers so generated SQL and telemetry records align
         * with the lab infrastructure.
         *
         * @param project_id Google Cloud project ID.
         * @param region BigQuery dataset and connection region.
         * @param dataset Target BigQuery dataset ID (default `"bwg"`).
         * @param connection_name BigQuery Cloud Resource connection name.
         * @param client Optional `google.cloud.bigquery.Client` instance for DI.
         */
        """
        cfg = get_config()
        self.project_id = (project_id or cfg.project_id or "local-dev-project").strip()
        self.region = (region or cfg.region or "us-central1").strip()
        self.dataset = dataset.strip() or "bwg"
        self.connection_name = connection_name.strip() or "pitch-connection"
        if client is not None:
            self.client = client
        elif _is_live_cloud_configured(project_id=self.project_id):
            try:
                from google.cloud import bigquery

                self.client = bigquery.Client(project=self.project_id)
            except Exception:
                self.client = None
        else:
            self.client = None
        self.events: list[dict[str, Any]] = []
        self._telemetry_events = self.events

    def record_telemetry(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        /**
         * Record an agent execution telemetry event.
         *
         * Why: Captures structured execution metadata (session ID, active node, latency,
         * routing decision, token usage) in memory and forwards to BigQuery when a live
         * client is injected.
         *
         * @param event Non-empty telemetry dictionary.
         * @return Enriched telemetry event dictionary with timestamp and dataset info.
         */
        """
        if not isinstance(event, dict) or not event:
            raise ValueError("telemetry event must be a non-empty dictionary")
        enriched = copy.deepcopy(event)
        enriched.setdefault("project_id", self.project_id)
        enriched.setdefault("dataset", self.dataset)
        enriched.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        self.events.append(enriched)
        if self.client is not None and hasattr(self.client, "insert_rows_json"):
            table_id = f"{self.project_id}.{self.dataset}.agent_events"
            # Format row to align with bwg.agent_events schema:
            # session_id (STRING), agent_name (STRING), latency (FLOAT), token_usage (INTEGER), event_type (STRING), recorded_at (STRING)
            row_to_insert = {
                "session_id": str(enriched.get("session_id") or "default"),
                "agent_name": str(enriched.get("agent_name") or enriched.get("agent") or "pitch_generator"),
                "latency": float(enriched.get("latency_ms", enriched.get("latency", 0.0))),
                "token_usage": int(enriched.get("tokens", enriched.get("token_usage", 0))),
                "event_type": str(enriched.get("status") or enriched.get("event_type") or "completed"),
                "recorded_at": str(enriched.get("recorded_at")),
            }
            try:
                self.client.insert_rows_json(table_id, [row_to_insert])
            except Exception:
                pass
        return copy.deepcopy(enriched)

    def register_key_visuals(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Validate and register key visual records destined for BigQuery Object Tables.
         *
         * Why: Ensures records are non-empty and every image URI strictly begins with gs://
         * so BigQuery OBJ.MAKE_REF queries can resolve the unstructured Cloud Storage blobs.
         *
         * @param records List of visual record dicts containing campaign, concept, and uri.
         * @return Validated defensive copy of records.
         */
        """
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a non-empty list of dictionaries")
        validated: list[dict[str, Any]] = []
        for r in records:
            if not isinstance(r, dict):
                raise ValueError("Each record must be a dictionary")
            uri = str(r.get("uri", "")).strip()
            if not uri.startswith("gs://"):
                raise ValueError(f"Every key visual record must have a 'gs://' URI, got: {uri!r}")
            validated.append(copy.deepcopy(r))
        return validated

    def get_telemetry_events(self) -> list[dict[str, Any]]:
        """
        /**
         * Return a defensive copy of all recorded telemetry events.
         *
         * Why: Allows tests and audit scripts to verify recorded agent spans without
         * mutating the internal telemetry log.
         *
         * @return List of recorded telemetry event dictionaries.
         */
        """
        return copy.deepcopy(self.events)

    def build_key_visuals_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        bucket_name: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        campaigns: list[dict[str, str]] | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.key_visuals` with Cloud
         * Storage `ObjectRef` columns (`OBJ.MAKE_REF` and `OBJ.FETCH_METADATA`).
         *
         * Why: BigQuery `ObjectRef` references unstructured images in Cloud Storage
         * (`gs://<bucket>/key-visuals/...`) via the delegated Cloud Resource connection
         * (`<region>.pitch-connection`) so multimodal SQL functions (`AI.SCORE`) can
         * inspect key visuals directly in SQL (`en.md` L1237-1258).
         *
         * @param project_id Google Cloud project ID.
         * @param region Connection region (e.g., `"us-central1"`).
         * @param bucket_name Cloud Storage bucket storing key visual PNGs.
         * @param dataset BigQuery dataset name (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param campaigns Optional list of `{"campaign": ..., "concept": ..., "uri": ...}`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_project = (project_id or self.project_id).strip()
        resolved_region = (region or self.region).strip()
        resolved_bucket = normalize_bucket_name(bucket_name or f"{resolved_project}-bwg")
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()

        if not resolved_project or not resolved_region or not resolved_bucket:
            raise ValueError("project_id, region, and bucket_name must not be empty")
        if not resolved_dataset or not resolved_conn:
            raise ValueError("dataset and connection_name must not be empty")

        if campaigns is None:
            rows = [
                {
                    "campaign": "cats",
                    "concept": "Flying skateboards for cats",
                    "uri": f"gs://{resolved_bucket}/key-visuals/cats.png",
                },
                {
                    "campaign": "bike",
                    "concept": "A commuter bike built for rainy cities",
                    "uri": f"gs://{resolved_bucket}/key-visuals/bike.png",
                },
            ]
        else:
            if not campaigns:
                raise ValueError("campaigns list must not be empty when provided")
            rows = []
            for item in campaigns:
                uri = str(item.get("uri", "")).strip()
                if not uri.startswith("gs://"):
                    raise ValueError(f"Campaign image URI must start with 'gs://', got {uri!r}")
                rows.append(
                    {
                        "campaign": str(item.get("campaign", "campaign")).strip(),
                        "concept": str(item.get("concept", "")).strip(),
                        "uri": uri,
                    }
                )

        struct_lines = []
        for idx, row in enumerate(rows):
            camp_esc = row["campaign"].replace("'", "\\'")
            conc_esc = row["concept"].replace("'", "\\'")
            uri_esc = row["uri"].replace("'", "\\'")
            if idx == 0:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}' AS campaign, '{conc_esc}' AS concept,\n"
                    f"         '{uri_esc}' AS uri)"
                )
            else:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}', '{conc_esc}',\n"
                    f"         '{uri_esc}')"
                )

        structs_sql = ",\n".join(struct_lines)
        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.key_visuals AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  OBJ.FETCH_METADATA(\n"
            f"    OBJ.MAKE_REF(uri, '{resolved_region}.{resolved_conn}')\n"
            f"  ) AS key_visual\n"
            f"FROM UNNEST([\n"
            f"{structs_sql}\n"
            f"]);\n"
        )

    def build_brand_score_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        model_endpoint: str | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.brand_review` using `AI.SCORE`.
         *
         * Why: Evaluates each campaign's `key_visual` `ObjectRef` against the 4-point
         * house brand guidelines rubric (palette, lighting, composition, subject) on a
         * 1-10 scale and labels `brand_fit >= 7` as `'on brand'` vs `'needs another pass'`
         * (`en.md` L1311-1336).
         *
         * @param project_id Google Cloud project ID.
         * @param region Cloud Resource connection region.
         * @param dataset BigQuery dataset ID (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param model_endpoint Optional Gemini model endpoint name for `AI.SCORE`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_region = (region or self.region).strip()
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()
        endpoint = (model_endpoint or DEFAULT_FLASH_MODEL).strip()

        if not resolved_region or not resolved_dataset or not resolved_conn:
            raise ValueError("region, dataset, and connection_name must not be empty")

        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.brand_review AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  brand_fit,\n"
            f"  IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict\n"
            f"FROM (\n"
            f"  SELECT\n"
            f"    campaign,\n"
            f"    concept,\n"
            f"    AI.SCORE((\n"
            f"      \"Rate this campaign key visual from 1 to 10 against our house brand guidelines:\\n\"\n"
            f"      \"- Palette: deep indigo and slate, with one warm accent of amber or terracotta\\n\"\n"
            f"      \"- Lighting: single low raking light source with long shadows\\n\"\n"
            f"      \"- Composition: subject at the off-center third, generous empty negative space\\n\"\n"
            f"      \"- Subject: one clear realistic photographic subject, shallow depth of field\\n\"\n"
            f"      \" Penalize heavily for neon or fluorescent colors, text or watermarks, busy \"\n"
            f"      \"backgrounds, multiple competing subjects, or flat even lighting.\\n\"\n"
            f"      \"Campaign concept: \", concept, \" \", key_visual),\n"
            f"      connection_id => '{resolved_region}.{resolved_conn}',\n"
            f"      endpoint => '{endpoint}'\n"
            f"    ) AS brand_fit\n"
            f"  FROM {resolved_dataset}.key_visuals\n"
            f");\n"
        )

    def score_brand_compliance(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Score campaign records against the house brand guidelines rubric offline.
         *
         * Why: Mirrors BigQuery `AI.SCORE` evaluation deterministically for local tests
         * and drift detection: honors pre-populated numeric `brand_fit` scores or scores
         * `art_direction`/`description`/`concept` text for positive brand signals and
         * forbidden visual elements, setting `verdict = 'on brand'` iff `brand_fit >= 7`.
         *
         * @param records List of campaign record dictionaries to evaluate.
         * @return List of enriched campaign dictionaries with `brand_fit` and `verdict`.
         */
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list of dictionaries")

        evaluated: list[dict[str, Any]] = []
        for rec in records:
            item = copy.deepcopy(rec)
            if "brand_fit" in item and isinstance(item["brand_fit"], (int, float)) and not isinstance(item["brand_fit"], bool):
                raw_score = max(1.0, min(10.0, float(item["brand_fit"])))
                item["brand_fit"] = raw_score
            else:
                text_parts = [
                    str(item.get("art_direction", "")),
                    str(item.get("description", "")),
                    str(item.get("prompt", "")),
                    str(item.get("concept", "")),
                ]
                combined = " ".join(text_parts).lower()
                score = 3.0
                matched_categories: list[str] = []
                for cat_name, keywords in self.POSITIVE_BRAND_SIGNALS:
                    if any(kw in combined for kw in keywords):
                        score += 1.5
                        matched_categories.append(cat_name)

                penalties: list[str] = []
                for forbidden in self.FORBIDDEN_BRAND_TERMS:
                    if forbidden in combined:
                        score -= 3.0
                        penalties.append(forbidden)

                raw_score = max(1.0, min(10.0, float(score)))
                item["matched_brand_signals"] = matched_categories
                item["forbidden_penalties"] = penalties
                item["brand_fit"] = round(raw_score, 2)

            item["verdict"] = "on brand" if raw_score >= 7.0 else "needs another pass"
            evaluated.append(item)
        return evaluated


# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]
# TODO (Step 2a): Add key visual registration and save the BigQuery Object Table SQL script:
#   - Create `pitch_generator/sql/create_key_visuals.sql` containing a `CREATE OR REPLACE TABLE bwg.key_visuals AS`
#     query that wraps Cloud Storage `gs://` URIs in `OBJ.FETCH_METADATA(OBJ.MAKE_REF(uri, '<region>.pitch-connection')) AS key_visual`.
#   - Define `register_key_visuals(records: list[dict[str, Any]], service: BigQueryAnalyticsService | None = None) -> list[dict[str, Any]]`
#     (and method on `BigQueryAnalyticsService`) validating that `records` is non-empty and every `"uri"` starts with `"gs://"`,
#     raising `ValueError` on empty or non-`gs://` URIs, and delegate `build_key_visuals_sql(...)` at module level.


def register_key_visuals(
    records: list[dict[str, Any]],
    service: BigQueryAnalyticsService | None = None,
) -> list[dict[str, Any]]:
    """
    /**
     * Validate and register key visual records for BigQuery analytics.
     *
     * Why: Validates that visual records are non-empty and all URIs use Cloud Storage gs:// scheme
     * so BigQuery OBJ.MAKE_REF and OBJ.FETCH_METADATA functions can query unstructured assets.
     *
     * @param records List of campaign visual metadata dictionaries.
     * @param service Optional BigQueryAnalyticsService instance.
     * @return Validated list of visual records.
     */
    """
    svc = service or BigQueryAnalyticsService()
    return svc.register_key_visuals(records)


def build_key_visuals_sql(
    project_id: str | None = None,
    region: str | None = None,
    bucket_name: str | None = None,
    dataset: str = "bwg",
    connection_name: str = "pitch-connection",
    campaigns: list[dict[str, str]] | None = None,
) -> str:
    """
    /**
     * Construct BigQuery DDL query creating bwg.key_visuals using OBJ.MAKE_REF and OBJ.FETCH_METADATA.
     *
     * Why: Delegates to BigQueryAnalyticsService.build_key_visuals_sql so callers can generate
     * Cloud Resource connection Object Table queries without instantiating a service manually.
     */
    """
    svc = BigQueryAnalyticsService()
    return svc.build_key_visuals_sql(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        dataset=dataset,
        connection_name=connection_name,
        campaigns=campaigns,
    )


# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]
# TODO (Step 2b): Add multimodal brand drift detection and automated prompt/skill tuning:
#   - Create `pitch_generator/sql/score_brand_fit.sql` containing a `CREATE OR REPLACE TABLE bwg.brand_review AS`
#     query using `AI.SCORE((..., concept, key_visual), connection_id => '<region>.pitch-connection') AS brand_fit`
#     and computing `IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict`.
#   - Define module-level `score_brand_compliance(records: list[dict[str, Any]]) -> list[dict[str, Any]]`
#     labeling `brand_fit >= 7.0` as `'on brand'` and `< 7.0` as `'needs another pass'`.
#   - Define `detect_brand_drift(records: list[dict[str, Any]], threshold: float = 7.0) -> dict[str, Any]`
#     returning `"has_drift"` (`bool`), `"drifted_records"`, `"compliant_records"`, and `"drift_rate"`.
#   - Define `tune_prompt_and_skill(record: dict[str, Any]) -> dict[str, Any]` that strips forbidden off-brand
#     terms (`neon`, `cyan`, `magenta`, `watermark`, `logo`, `ring light`, `3d render`) and appends the 4 house
#     brand pillars (`deep indigo and slate with warm amber accent`, `single low raking light with long shadows`,
#     `off-center composition with generous negative space`, `one realistic photographic subject`) so the re-scored
#     `brand_fit >= 7.0` and `verdict == "on brand"`.


def score_brand_compliance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    /**
     * Score a list of campaign records against the house brand guidelines rubric.
     *
     * Why: Provides a module-level helper labeling brand_fit >= 7.0 as 'on brand'
     * and < 7.0 as 'needs another pass'.
     */
    """
    svc = BigQueryAnalyticsService()
    return svc.score_brand_compliance(records)


def detect_brand_drift(
    records: list[dict[str, Any]],
    threshold: float = 7.0,
) -> dict[str, Any]:
    """
    /**
     * Detect drifted visual prompts and outputs scoring below the brand rubric threshold.
     *
     * Why: Identifies campaigns needing automated prompt and skill tuning ('needs another pass')
     * to prevent subtle brand degradation across generative image iterations.
     *
     * @param records List of campaign visual metadata dictionaries.
     * @param threshold Cutoff score (default 7.0) below which an output is considered drifted.
     * @return Summary dict with has_drift, drifted_records, compliant_records, and drift_rate.
     */
    """
    scored = score_brand_compliance(records)
    drifted: list[dict[str, Any]] = []
    compliant: list[dict[str, Any]] = []

    for r in scored:
        score = float(r.get("brand_fit", 0.0))
        if score < threshold or r.get("verdict") == "needs another pass":
            drifted.append(r)
        else:
            compliant.append(r)

    total = len(scored)
    drift_rate = round(len(drifted) / total, 2) if total > 0 else 0.0

    return {
        "has_drift": len(drifted) > 0,
        "drifted_records": drifted,
        "compliant_records": compliant,
        "drift_rate": drift_rate,
    }


def tune_prompt_and_skill(record: dict[str, Any]) -> dict[str, Any]:
    """
    /**
     * Automatically repair and tune an off-brand prompt or art direction record.
     *
     * Why: Implements closed-loop drift remediation: strips forbidden off-brand terms
     * (neon, cyan, magenta, watermark, logo, ring light, 3d render) and reinforces the four
     * house brand aesthetic pillars so re-scored outputs reliably reach >= 7.0 ('on brand').
     *
     * @param record Campaign dictionary containing prompt, art_direction, or concept.
     * @return Tuned campaign dictionary with updated text, brand_fit >= 7.0, and verdict == 'on brand'.
     */
    """
    tuned = copy.deepcopy(record)
    # Clean forbidden terms across all text fields
    for k in ("prompt", "art_direction", "concept", "description"):
        if k in tuned and isinstance(tuned[k], str):
            text = tuned[k]
            for pat in [
                r"\bneon\b",
                r"\bfluorescent\b",
                r"\bcyan\b",
                r"\bmagenta\b",
                r"\bwatermark(s)?\b",
                r"\blogo(s)?\b",
                r"\bring\s+light\b",
                r"\b(3d\s+render|cgi|octane\s+render)\b",
                r"\bchrome\b",
                r"\blens\s+flare\b",
                r"\bbokeh\b",
                r"\bsparkle\b",
                r"\bfisheye\b",
            ]:
                text = re.sub(pat, "", text, flags=re.IGNORECASE)
            tuned[k] = re.sub(r"\s+", " ", text).strip()

    # Append house pillars to primary text key
    text_key = "prompt" if "prompt" in tuned else ("art_direction" if "art_direction" in tuned else "concept")
    house_pillars_addition = (
        " Palette: deep indigo and slate with warm amber accent. "
        "Lighting: single low raking light with long shadows. "
        "Composition: off-center composition with generous negative space. "
        "Subject: one realistic photographic subject, shallow depth of field."
    )
    current_val = tuned.get(text_key, "")
    tuned[text_key] = f"{current_val}.{house_pillars_addition}" if current_val else house_pillars_addition.strip()

    # Clear pre-existing brand_fit if any
    tuned.pop("brand_fit", None)
    tuned.pop("verdict", None)

    # Re-evaluate brand compliance
    re_scored = score_brand_compliance([tuned])[0]
    return re_scored


# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]
# TODO (Step 3b): Implement recursive PII redaction across strings, dicts, and lists before model or telemetry calls:
#   - `PII_PATTERNS`: ordered regex list matching API secrets (`[REDACTED_SECRET]`), emails (`[REDACTED_EMAIL]`),
#     credit cards (`[REDACTED_CC]`), SSNs (`[REDACTED_SSN]`), and phone numbers (`[REDACTED_PHONE]`) without
#     corrupting normal ISO dates (`2026-10-15`) or campaign metrics (`42%`, `350 lumen`).
#   - `ScrubResult(str)` (or dataclass/str hybrid) exposing `.scrubbed` and `.redaction_count: int` while
#     supporting substring checks (`"[REDACTED_EMAIL]" in res`).
#   - `PIIScrubber` class with `.scrub_text(text) -> ScrubResult` and `.scrub_payload(payload) -> ScrubResult`
#     (deep-copying dicts/lists so caller inputs are never mutated in place), plus module-level helpers
#     `scrub_pii`, `scrub_text`, and `scrub_payload`.

PII_PATTERNS: tuple[tuple[str, re.Pattern, str], ...] = (
    # API Secrets / Keys (Google API keys, Bearer tokens, general high-entropy secrets)
    ("secret", re.compile(r"\bAIza[0-9A-Za-z-_]{35}\b|\b(?:api[_-]?key|secret|token)[\s:=]+['\"]?([A-Za-z0-9_\-]{16,})['\"]?", re.IGNORECASE), "[REDACTED_SECRET]"),
    # Email addresses
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Credit cards (16 digits with dashes or spaces)
    ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[REDACTED_CC]"),
    # US Social Security Numbers (SSN: 3-2-4 digits)
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # Phone numbers (US/Intl with area codes, avoiding simple dates)
    ("phone", re.compile(r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED_PHONE]"),
)


class ScrubResult(str):
    """
    /**
     * String subclass containing sanitized content and redaction statistics.
     *
     * Why: Preserves direct string comparability (`"[REDACTED_EMAIL]" in res` or `str(res)`)
     * while exposing `.scrubbed` and `.redaction_count` metadata.
     */
    """

    def __new__(cls, content: str, redaction_count: int = 0):
        obj = super().__new__(cls, content)
        obj.scrubbed = content
        obj.redaction_count = redaction_count
        return obj


class PIIScrubber:
    """
    /**
     * Recursive PII redaction engine protecting telemetry and model calls from data leakage.
     *
     * Why: Sanitizes customer contact info, SSNs, credit cards, and API secrets across
     * unstructured text strings and complex nested JSON structures while preserving dates and metrics.
     */
    """

    def __init__(self, patterns: Sequence[tuple[str, re.Pattern, str]] | None = None) -> None:
        self.patterns = tuple(patterns) if patterns is not None else PII_PATTERNS

    def scrub_text(self, text: str) -> ScrubResult:
        if not isinstance(text, str) or not text:
            return ScrubResult(str(text or ""), 0)

        cleaned = text
        total_redactions = 0

        # Run ordered redactions
        for name, pattern, replacement in self.patterns:
            matches = list(pattern.finditer(cleaned))
            if matches:
                total_redactions += len(matches)
                cleaned = pattern.sub(replacement, cleaned)

        return ScrubResult(cleaned, total_redactions)

    def scrub_payload(self, payload: Any) -> Any:
        # Deepcopy to prevent mutating caller inputs
        cloned = copy.deepcopy(payload)
        _, scrubbed_obj = self._scrub_recursive(cloned)
        return scrubbed_obj

    def _scrub_recursive(self, obj: Any) -> tuple[int, Any]:
        count = 0
        if isinstance(obj, str):
            res = self.scrub_text(obj)
            return res.redaction_count, res
        elif isinstance(obj, dict):
            out_dict = {}
            for k, v in obj.items():
                sub_count, sub_val = self._scrub_recursive(v)
                count += sub_count
                out_dict[k] = sub_val
            return count, out_dict
        elif isinstance(obj, (list, tuple)):
            out_list = []
            for item in obj:
                sub_count, sub_val = self._scrub_recursive(item)
                count += sub_count
                out_list.append(sub_val)
            return count, type(obj)(out_list)
        return 0, obj


_GLOBAL_PII_SCRUBBER = PIIScrubber()


def scrub_text(text: str) -> ScrubResult:
    """
    /**
     * Module-level helper scrubbing sensitive PII from a single string.
     */
    """
    return _GLOBAL_PII_SCRUBBER.scrub_text(text)


def scrub_payload(payload: Any) -> Any:
    """
    /**
     * Module-level helper recursively scrubbing sensitive PII from dicts/lists without mutating inputs.
     */
    """
    return _GLOBAL_PII_SCRUBBER.scrub_payload(payload)


def scrub_pii(text_or_payload: Any) -> Any:
    """
    /**
     * Universal module-level helper scrubbing PII from text or complex payloads.
     */
    """
    if isinstance(text_or_payload, str):
        return scrub_text(text_or_payload)
    return scrub_payload(text_or_payload)



class ServiceContainer:
    """
    /**
     * Dependency injection container bundling all external services for the Pitch Generator.
     *
     * Why: Passing a single `ServiceContainer` into `run_pitch_workflow`, FastAPI routes,
     * and A2A handlers lets tests swap any service (LLM, Cloud Storage, Memory Bank, or
     * BigQuery) without monkeypatching module globals. Exposes both `.llm`/`.llm_client`
     * and `.artifacts`/`.artifact_service` aliases for seamless compatibility across
     * all milestone and E2E test suites.
     */
    """

    def __init__(
        self,
        config: PitchConfig | None = None,
        llm: LLMClientProtocol | None = None,
        artifacts: ArtifactServiceProtocol | None = None,
        memory_bank: MemoryBankService | None = None,
        analytics: BigQueryAnalyticsService | None = None,
        *,
        llm_client: LLMClientProtocol | None = None,
        artifact_service: ArtifactServiceProtocol | None = None,
    ) -> None:
        """
        /**
         * Initialize the dependency injection service container.
         *
         * Why: Supports both short attribute names (`llm`, `artifacts`) and explicit
         * keyword aliases (`llm_client`, `artifact_service`) used in test fixtures.
         *
         * @param config Resolved `PitchConfig` (defaults to `get_config()`).
         * @param llm Injected `LLMClientProtocol` instance.
         * @param artifacts Injected `ArtifactServiceProtocol` instance.
         * @param memory_bank Injected `MemoryBankService` instance.
         * @param analytics Injected `BigQueryAnalyticsService` instance.
         * @param llm_client Optional alias for `llm`.
         * @param artifact_service Optional alias for `artifacts`.
         */
        """
        self.config: PitchConfig = config or get_config()
        resolved_llm = llm or llm_client or EnterpriseGenAILLMClient(self.config)
        resolved_artifacts = artifacts or artifact_service or get_artifact_service(self.config)
        self.llm: LLMClientProtocol = resolved_llm
        self.llm_client: LLMClientProtocol = resolved_llm
        self.artifacts: ArtifactServiceProtocol = resolved_artifacts
        self.artifact_service: ArtifactServiceProtocol = resolved_artifacts
        self.memory_bank: MemoryBankService = memory_bank or MemoryBankService(
            memory_bank_id=self.config.memory_bank_id
        )
        self.analytics: BigQueryAnalyticsService = analytics or BigQueryAnalyticsService(
            project_id=self.config.project_id,
            region=self.config.region,
        )


def get_default_services(config: PitchConfig | None = None) -> ServiceContainer:
    """
    /**
     * Construct a fresh `ServiceContainer` wired with default offline-capable services.
     *
     * Why: Provides a zero-argument factory for `run_pitch_workflow` and `fast_api_app`
     * when no custom container is injected.
     *
     * @param config Optional `PitchConfig` override.
     * @return Initialized `ServiceContainer`.
     */
    """
    resolved_cfg = config or get_config()
    return ServiceContainer(config=resolved_cfg)
