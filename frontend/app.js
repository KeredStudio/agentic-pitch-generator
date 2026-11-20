"use strict";

/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file app.js
 * @description Vanilla ES6 JavaScript client for the Agentic Pitch Generator web UI.
 *
 * Why: Connects the plain HTML5 interface to the FastAPI backend endpoints
 * (`/api/health`, `/api/config`, `/api/pitch`, `/api/approve`, and
 * `/a2a/pitch_generator/.well-known/agent-card.json`) without third-party UI libraries.
 * Uses strictly safe DOM APIs (`document.createElement`, `textContent`, `replaceChildren`)
 * to prevent DOM-based cross-site scripting.
 */

/**
 * Create a DOM element with safe attribute assignment and plain text content.
 *
 * Why: Constructing elements exclusively via `document.createElement` and `textContent`
 * guarantees that untrusted model outputs or user briefs are never parsed as executable
 * HTML markup.
 *
 * @param {string} tagName HTML tag name to create.
 * @param {Object<string, string>} attributes Key-value map of element attributes.
 * @param {string} textValue Optional text content assigned via `textContent`.
 * @return {HTMLElement} Newly created DOM element.
 */
function createSafeElement(tagName, attributes, textValue) {
  const el = document.createElement(tagName);
  if (attributes && typeof attributes === "object") {
    Object.keys(attributes).forEach((key) => {
      if (attributes[key] !== undefined && attributes[key] !== null) {
        el.setAttribute(key, String(attributes[key]));
      }
    });
  }
  if (textValue !== undefined && textValue !== null) {
    el.textContent = String(textValue);
  }
  return el;
}

/**
 * Append inline Markdown tokens (`**bold**`, `*italic*`) to a parent DOM element
 * using safe DOM APIs (`createTextNode`, `createSafeElement`).
 *
 * Why: Renders inline emphasis cleanly while guaranteeing that raw HTML or script
 * tags in model output are always treated as literal text nodes (never `innerHTML`).
 *
 * @param {HTMLElement} parentEl Target parent DOM element.
 * @param {string} lineText Single line of text with code spans already stripped.
 * @return {void}
 */
function appendInlineMarkdown(parentEl, lineText) {
  const tokenPattern = /(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_)/g;
  let lastIndex = 0;
  let match = tokenPattern.exec(lineText);
  while (match !== null) {
    if (match.index > lastIndex) {
      parentEl.appendChild(
        document.createTextNode(lineText.slice(lastIndex, match.index))
      );
    }
    const token = match[0];
    if (
      (token.startsWith("**") && token.endsWith("**")) ||
      (token.startsWith("__") && token.endsWith("__"))
    ) {
      parentEl.appendChild(createSafeElement("strong", {}, token.slice(2, -2)));
    } else {
      parentEl.appendChild(createSafeElement("em", {}, token.slice(1, -1)));
    }
    lastIndex = tokenPattern.lastIndex;
    match = tokenPattern.exec(lineText);
  }
  if (lastIndex < lineText.length) {
    parentEl.appendChild(document.createTextNode(lineText.slice(lastIndex)));
  }
}

/**
 * Render basic Markdown (headings, bold, italics, lists, paragraphs, line breaks)
 * into `containerEl` while stripping fenced/inline code and avoiding `innerHTML`.
 *
 * @param {HTMLElement} containerEl Target container DOM element.
 * @param {string} rawMarkdown Raw Markdown string from an agent.
 * @return {void}
 */
function renderSafeMarkdown(containerEl, rawMarkdown) {
  if (!containerEl) {
    return;
  }
  const text = String(rawMarkdown || "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`([^`\n]+)`/g, "$1")
    .trim();

  if (
    typeof document === "undefined" ||
    typeof document.createTextNode !== "function" ||
    typeof containerEl.replaceChildren !== "function"
  ) {
    containerEl.textContent = text;
    return;
  }

  if (!text) {
    containerEl.replaceChildren();
    return;
  }

  const lines = text.split(/\r?\n/);
  const blocks = [];
  let currentParagraph = null;
  let currentList = null;
  let currentListType = null;

  const flushParagraph = () => {
    if (currentParagraph) {
      blocks.push(currentParagraph);
      currentParagraph = null;
    }
  };

  const flushList = () => {
    if (currentList) {
      blocks.push(currentList);
      currentList = null;
      currentListType = null;
    }
  };

  lines.forEach((rawLine) => {
    const trimmed = rawLine.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const hEl = createSafeElement("h4", { class: "md-heading" });
      appendInlineMarkdown(hEl, headingMatch[2].trim());
      blocks.push(hEl);
      return;
    }

    const ulMatch = trimmed.match(/^[-*•]\s+(.+)$/);
    if (ulMatch) {
      flushParagraph();
      if (currentListType !== "ul") {
        flushList();
        currentList = createSafeElement("ul", { class: "md-list" });
        currentListType = "ul";
      }
      const liEl = createSafeElement("li", {});
      appendInlineMarkdown(liEl, ulMatch[1].trim());
      currentList.appendChild(liEl);
      return;
    }

    const olMatch = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (olMatch) {
      flushParagraph();
      if (currentListType !== "ol") {
        flushList();
        currentList = createSafeElement("ol", { class: "md-list" });
        currentListType = "ol";
      }
      const liEl = createSafeElement("li", {});
      appendInlineMarkdown(liEl, olMatch[1].trim());
      currentList.appendChild(liEl);
      return;
    }

    flushList();
    if (!currentParagraph) {
      currentParagraph = createSafeElement("p", { class: "md-paragraph" });
    } else {
      currentParagraph.appendChild(createSafeElement("br", {}));
    }
    appendInlineMarkdown(currentParagraph, trimmed);
  });

  flushParagraph();
  flushList();
  containerEl.replaceChildren(...blocks);
}

/**
 * Display a non-blocking status message in `#status-banner`.
 *
 * Why: Avoids blocking browser modal dialogs and updates an ARIA live region
 * (`role="alert"`) for screen-reader accessibility.
 *
 * @param {string} message Notification message to display (empty string hides banner).
 * @param {string} level Severity indicator (`"info"`, `"warn"`, or `"error"`).
 * @return {void}
 */
function setBanner(message, level) {
  if (typeof document === "undefined") {
    return;
  }
  const banner = document.getElementById("status-banner");
  if (!banner) {
    return;
  }
  if (!message) {
    banner.hidden = true;
    banner.textContent = "";
    return;
  }
  banner.hidden = false;
  banner.setAttribute("data-level", level || "info");
  banner.textContent = String(message);
}

/**
 * Render execution trace steps in `#workflow-trace-list`.
 *
 * Why: Uses `replaceChildren` and `createSafeElement("li", ...)` so learners can
 * inspect which specialist nodes (`creative_director`, `copywriter`,
 * `brand_strategist`, `visual_director`, `assemble`, `package`) ran during the workflow.
 *
 * @param {Array<string>} traceSteps Ordered list of executed workflow node names.
 * @return {void}
 */
function renderWorkflowTrace(traceSteps) {
  if (typeof document === "undefined") {
    return;
  }
  const listEl = document.getElementById("workflow-trace-list");
  if (!listEl) {
    return;
  }
  const steps =
    Array.isArray(traceSteps) && traceSteps.length > 0
      ? traceSteps
      : [
          "creative_director",
          "copywriter",
          "brand_strategist",
          "visual_director",
          "assemble",
          "package",
        ];
  const items = steps.map((stepName) =>
    createSafeElement("li", {}, `Executed node: ${stepName}`)
  );
  listEl.replaceChildren(...items);
}

/**
 * Update the DOM cards in `#pitch-result-card` with the workflow response payload.
 *
 * Why: Populates Creative Director concept, Copywriter social copy, Brand Strategist
 * positioning, Visual Director art direction, Key Visual preview, and Human-in-the-Loop
 * approval gate state.
 *
 * @param {Object} data Parsed JSON response from `/api/pitch` or `/api/approve`.
 * @return {void}
 */
function renderPitchResult(data) {
  if (typeof document === "undefined" || !data) {
    return;
  }
  const statusEl = document.getElementById("pitch-status");
  const hitlCheckboxGroup = document.getElementById("hitl-checkbox-group");
  const hitlCard = document.getElementById("hitl-approval-card");
  const hitlPromptEl = document.getElementById("hitl-approval-prompt");
  const approveBtn = document.getElementById("btn-approve-concept");
  const rejectBtn = document.getElementById("btn-reject-concept");
  const conceptEl = document.getElementById("pitch-concept");
  const copyEl = document.getElementById("pitch-copy");
  const brandCard = document.getElementById("brand-strategy-card");
  const brandEl = document.getElementById("pitch-brand-strategy");
  const artCard = document.getElementById("art-direction-card");
  const artEl = document.getElementById("pitch-art-direction");
  const visualCard = document.getElementById("key-visual-card");
  const visualEl = document.getElementById("pitch-key-visual");
  const visualImgEl = document.getElementById("pitch-key-visual-img");

  const statusText = String(data.status || "completed");
  if (statusEl) {
    if (statusText === "completed") {
      statusEl.textContent = "✅ 生成完成 (Completed)";
      statusEl.className = "badge badge-ok";
    } else if (statusText === "input_required") {
      statusEl.textContent = "⏸️ 等待人工審核 (Paused at HITL)";
      statusEl.className = "badge badge-warn";
    } else if (statusText === "rejected") {
      statusEl.textContent = "🛑 已駁回中止 (Rejected)";
      statusEl.className = "badge badge-danger";
    } else {
      statusEl.textContent = statusText;
      statusEl.className = "badge badge-neutral";
    }
  }
  const showHitlCard =
    Boolean(data.hitl_enabled) ||
    statusText === "input_required" ||
    statusText === "rejected";
  if (hitlCheckboxGroup) {
    hitlCheckboxGroup.hidden = !showHitlCard;
  }
  if (hitlCard) {
    hitlCard.hidden = !showHitlCard;
    if (statusText === "input_required") {
      if (approveBtn) approveBtn.disabled = false;
      if (rejectBtn) rejectBtn.disabled = false;
      if (hitlPromptEl) {
        hitlPromptEl.textContent =
          data.prompt ||
          "請審閱下方「創意總監核心概念」，若符合預期請點選「同意此概念」，若不滿意可「駁回此概念」。";
      }
      setBanner(
        "【人機協同 (HITL) 審核卡點】：創意總監已完成概念發想，請審閱下方內容並點擊審核按鈕。",
        "info"
      );
    } else {
      if (approveBtn) approveBtn.disabled = true;
      if (rejectBtn) rejectBtn.disabled = true;
      if (data.hitl_requested && data.hitl_enabled === false) {
        if (hitlPromptEl) {
          hitlPromptEl.textContent =
            "您勾選了人工介入審批，但後端尚未實作 approve_concept，流程已自動直通執行。";
        }
        setBanner(
          "人工介入閘門尚未實作，流程直接執行完畢。",
          "warn"
        );
      } else if (statusText === "rejected") {
        if (hitlPromptEl) {
          hitlPromptEl.textContent =
            "行銷概念已被審核員駁回。下游所有代理人與視覺生成均已安全中止，節省 100% 後續 Token。";
        }
        setBanner(
          "提案概念已駁回。工作流程已安全停止，未消耗下游任何資源。",
          "warn"
        );
      } else {
        if (hitlPromptEl) {
          hitlPromptEl.textContent = data.hitl_enabled
            ? "HITL 人機協同審核閘門已就緒。勾選「啟用人工介入審批」後點擊生成，系統將在概念發想後自動暫停供您確認。"
            : "尚未啟用 HITL 審批閘門。";
        }
        setBanner("", "info");
      }
    }
  }
  if (conceptEl) {
    renderSafeMarkdown(conceptEl, data.concept || "尚未產生概念。");
  }
  if (copyEl) {
    const defaultCopyMessage =
      statusText === "input_required"
        ? "Paused at HITL gate — awaiting human approval before running downstream agents."
        : statusText === "rejected"
          ? "Workflow halted — concept rejected by human reviewer (zero downstream tokens spent)."
          : "No social copy generated.";
    renderSafeMarkdown(copyEl, data.copy || defaultCopyMessage);
  }
  if (brandCard && brandEl) {
    brandCard.hidden = false;
    const defaultBrandMessage =
      statusText === "input_required"
        ? "⏸️ 已在 HITL 閘門暫停 — 等待人工確認概念。"
        : statusText === "rejected"
          ? "🛑 流程已中止 — 概念遭退件駁回。"
          : "尚未產生品牌策略。";
    renderSafeMarkdown(brandEl, data.brand_strategy || defaultBrandMessage);
  }
  if (artCard && artEl) {
    artCard.hidden = false;
    const defaultArtMessage =
      statusText === "input_required"
        ? "⏸️ 已在 HITL 閘門暫停 — 等待人工確認概念。"
        : statusText === "rejected"
          ? "🛑 流程已中止 — 概念遭退件駁回。"
          : "尚未產生藝術指導指南。";
    renderSafeMarkdown(artEl, data.art_direction || defaultArtMessage);
  }
  if (visualCard && visualEl) {
    visualCard.hidden = false;
    if (data.key_visual_uri) {
      const rawUri = String(data.key_visual_uri);
      const sessionId = encodeURIComponent(
        String(data.session_id || "default").trim() || "default"
      );
      const resolvedImgUrl =
        data.key_visual_url ||
        (rawUri.startsWith("gs://")
          ? `/api/artifacts/${sessionId}/key_visual.png`
          : rawUri);
      if (visualImgEl) {
        visualImgEl.src = resolvedImgUrl;
        visualImgEl.hidden = false;
      }
      visualEl.textContent = `產物存放位置 (GCS Artifact URI): ${rawUri}`;
    } else {
      if (visualImgEl) {
        visualImgEl.removeAttribute("src");
        visualImgEl.hidden = true;
      }
      visualEl.textContent =
        statusText === "input_required"
          ? "⏸️ 已在 HITL 閘門暫停 — 等待人工審查概念。"
          : statusText === "rejected"
            ? "🛑 流程已中止 — 不會呼叫 Imagen 生成圖像。"
            : "等待生成品牌主視覺圖像...";
    }
  }
  renderWorkflowTrace(data.trace);

  const a2aFlowEl = document.getElementById("a2a-flow-summary");
  if (a2aFlowEl) {
    if (statusText === "completed") {
      a2aFlowEl.textContent =
        `[A2A 跨代理人通訊協議：已成功執行 (ACTIVE)]\n` +
        `• 發起端 (Initiator)：Pitch Generator 主工作流程根代理人\n` +
        `• 目標端 (Target)：遠端 Visual Director 微服務 (http://127.0.0.1:8801/)\n` +
        `• 通訊標準 (Protocol)：基於 HTTP 之 JSON-RPC 2.0 (已驗證 Agent Card 通訊名片)\n` +
        `• 傳遞內容 (Context Passed)：創意概念 (Concept) + 品牌策略定位 (Brand Strategy)\n` +
        `• 視覺總監動作 (Actions)：加載 brand-guidelines 專業 Skill -> 調用 generate_key_visual 工具\n` +
        `• 最終產物 (Result)：成功回傳高畫質 Key Visual 影像並呈現於上方！`;
      a2aFlowEl.style.whiteSpace = "pre-line";
    } else if (statusText === "input_required") {
      a2aFlowEl.textContent =
        `[A2A 跨代理人通訊協議：安全掛起中 (PAUSED)]\n` +
        `• 人工審核 (HITL) 閘門作動中。對遠端 Visual Director 的 A2A 跨網路呼叫已暫停，等待您的審批以節省雲端成本。`;
      a2aFlowEl.style.whiteSpace = "pre-line";
    } else if (statusText === "rejected") {
      a2aFlowEl.textContent =
        `[A2A 跨代理人通訊協議：安全終止 (HALTED)]\n` +
        `• 概念已被駁回。系統零網路請求發送至 Visual Director，無消耗任何額外圖片生成與下游模型 Token。`;
      a2aFlowEl.style.whiteSpace = "pre-line";
    }
  }
}

/**
 * Execute an HTTP request using the standard Fetch API (`fetch(url, options)`).
 *
 * Why: Provides a default `fetch(` implementation while allowing unit tests to inject
 * a custom `fetchImpl` mock when running offline outside a browser.
 *
 * @param {string} url Request URL or relative path.
 * @param {Object} [options] Optional Fetch API request options.
 * @return {Promise<Response>} Fetch API response promise.
 */
function defaultFetch(url, options) {
  return fetch(url, options);
}

/**
 * Fetch service health (`/api/health`) and public configuration (`/api/config`).
 *
 * Why: Verifies backend connectivity on page load and displays active project,
 * region, and Gemini Enterprise Agent Platform model in the header summary.
 *
 * @param {Function} fetchImpl Optional fetch implementation for dependency injection.
 * @return {Promise<Object>} Resolved health and config objects.
 */
async function fetchHealthAndConfig(fetchImpl) {
  const fetcher = fetchImpl || defaultFetch;
  const [healthResp, configResp] = await Promise.all([
    fetcher("/api/health"),
    fetcher("/api/config"),
  ]);
  const health = await healthResp.json();
  const cfg = await configResp.json();

  if (typeof document !== "undefined") {
    const healthBadge = document.getElementById("health-badge");
    if (healthBadge) {
      healthBadge.textContent = `服務狀態: ${health.status === "ok" ? "正常運作中 (OK)" : health.status}`;
      healthBadge.className = "badge badge-ok";
    }
    const summaryEl = document.getElementById("config-summary");
    if (summaryEl) {
      const modelName = (cfg.models && cfg.models.flash) || "gemini-3.8-flash";
      summaryEl.textContent =
        `專案代碼: ${cfg.project_id} | 雲端區域: ${cfg.region} | ` +
        `主力模型: ${modelName} | 企業級代理人架構: ${cfg.use_enterprise ? "已啟用" : "未啟用"}`;
    }
    const a2aBtn = document.getElementById("btn-inspect-a2a");
    if (a2aBtn && cfg.a2a_enabled) {
      a2aBtn.hidden = false;
    }
    const hitlCheckboxGroup = document.getElementById("hitl-checkbox-group");
    if (hitlCheckboxGroup && cfg.hitl_enabled) {
      hitlCheckboxGroup.hidden = false;
    }
    const hitlCard = document.getElementById("hitl-approval-card");
    if (hitlCard && cfg.hitl_enabled) {
      hitlCard.hidden = false;
    }
    const hitlPromptEl = document.getElementById("hitl-approval-prompt");
    if (hitlPromptEl && cfg.hitl_enabled) {
      hitlPromptEl.textContent =
        "HITL 閘門已就緒。勾選「啟用人工介入審批」後點擊生成，工作流程將在創意總監發想完成後暫停，等待您的審核。";
    }
  }
  return { health, config: cfg };
}

/**
 * Toggle the loading state and spinner on `#btn-generate-pitch`.
 *
 * Why: Disables the submit button and renders an accessible inline spinner while
 * the multi-agent workflow executes so users do not accidentally submit duplicate
 * requests and have clear visual feedback that work is in progress.
 *
 * @param {boolean} isLoading True while the pitch request is in flight.
 * @return {void}
 */
function setGenerateButtonLoading(isLoading) {
  if (typeof document === "undefined") {
    return;
  }
  const btn = document.getElementById("btn-generate-pitch");
  const statusEl = document.getElementById("pitch-status");
  if (!btn) {
    return;
  }
  if (isLoading) {
    btn.disabled = true;
    btn.setAttribute("aria-busy", "true");
    const spinnerEl = createSafeElement("span", {
      class: "spinner",
      "aria-hidden": "true",
    });
    const labelEl = createSafeElement("span", {}, "多代理人協同運算中...");
    btn.replaceChildren(spinnerEl, labelEl);
    if (statusEl) {
      statusEl.textContent = "⚙️ 代理人執行中 (Generating)...";
      statusEl.className = "badge badge-warn";
    }
  } else {
    btn.disabled = false;
    btn.removeAttribute("aria-busy");
    btn.textContent = "✨ 啟動多代理人生成提案";
  }
}

/**
 * Submit a campaign brief to `POST /api/pitch` and render the resulting pitch package.
 *
 * Why: Drives the primary campaign generation workflow using Gemini Enterprise Agent
 * Platform cloud models while locking the submit button until completion.
 *
 * @param {Object} payload Request dictionary with `brief`, `session_id`, and `require_approval`.
 * @param {Function} fetchImpl Optional fetch implementation for testing.
 * @return {Promise<Object>} Workflow execution response.
 */
async function submitPitchRequest(payload, fetchImpl) {
  if (typeof document !== "undefined") {
    const btn = document.getElementById("btn-generate-pitch");
    if (btn && btn.disabled) {
      return null;
    }
  }
  const fetcher = fetchImpl || fetch;
  setBanner("", "info");
  setGenerateButtonLoading(true);
  try {
    const resp = await fetcher("/api/pitch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      setBanner(data.error || "Failed to generate pitch.", "error");
      const statusEl =
        typeof document !== "undefined"
          ? document.getElementById("pitch-status")
          : null;
      if (statusEl) {
        statusEl.textContent = "error";
        statusEl.className = "badge badge-warn";
      }
      return data;
    }
    renderPitchResult(data);
    return data;
  } catch (err) {
    setBanner(`Failed to generate pitch: ${err.message || err}`, "error");
    const statusEl =
      typeof document !== "undefined"
        ? document.getElementById("pitch-status")
        : null;
    if (statusEl) {
      statusEl.textContent = "error";
      statusEl.className = "badge badge-warn";
    }
    throw err;
  } finally {
    setGenerateButtonLoading(false);
  }
}

/**
 * Submit a Human-in-the-Loop approval or rejection decision to `POST /api/approve`.
 *
 * Why: Resumes a paused workflow session (`status === "input_required"`) in Step 3c
 * so learners can test both concept rejection (fail-closed) and approval directly
 * from the web interface.
 *
 * @param {boolean} approved True to approve and run downstream agents, False to reject.
 * @param {Function} fetchImpl Optional fetch implementation for testing.
 * @return {Promise<Object>} Resumed workflow response payload.
 */
async function submitApprovalDecision(approved, fetchImpl) {
  const fetcher = fetchImpl || fetch;
  const sessionInput =
    typeof document !== "undefined"
      ? document.getElementById("session-id-input")
      : null;
  const approveBtn =
    typeof document !== "undefined"
      ? document.getElementById("btn-approve-concept")
      : null;
  const rejectBtn =
    typeof document !== "undefined"
      ? document.getElementById("btn-reject-concept")
      : null;
  const sessionId =
    sessionInput && sessionInput.value ? sessionInput.value.trim() : "default";

  if (approveBtn) approveBtn.disabled = true;
  if (rejectBtn) rejectBtn.disabled = true;
  setGenerateButtonLoading(true);
  try {
    const resp = await fetcher("/api/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        approved: Boolean(approved),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      setBanner(data.error || "Failed to process approval decision.", "error");
      if (approveBtn) approveBtn.disabled = false;
      if (rejectBtn) rejectBtn.disabled = false;
      return data;
    }
    renderPitchResult(data);
    return data;
  } catch (err) {
    setBanner(`Approval request failed: ${err.message || err}`, "error");
    if (approveBtn) approveBtn.disabled = false;
    if (rejectBtn) rejectBtn.disabled = false;
    throw err;
  } finally {
    setGenerateButtonLoading(false);
  }
}

/**
 * Fetch and display the Visual Director's well-known A2A Agent Card from
 * `/a2a/visual_director/.well-known/agent-card.json`.
 *
 * Why: Allows learners in Step 1c to inspect the published Visual Director A2A v0.3
 * discovery metadata (`/.well-known/agent-card.json`) directly in the web UI.
 *
 * @param {Function} fetchImpl Optional fetch implementation for testing.
 * @return {Promise<Object>} Parsed Visual Director A2A Agent Card JSON.
 */
async function inspectA2aAgentCard(fetchImpl) {
  const fetcher = fetchImpl || fetch;
  const resp = await fetcher("/a2a/visual_director/.well-known/agent-card.json");
  const card = await resp.json();
  if (typeof document !== "undefined") {
    const section = document.getElementById("a2a-card-section");
    const panel = document.getElementById("a2a-card-panel");
    if (section && panel) {
      section.hidden = false;
      panel.textContent = JSON.stringify(card, null, 2);
    }
  }
  return card;
}

/**
 * Attach DOM event listeners for form submission, HITL approval, and A2A Agent Card inspection.
 *
 * Why: Centralizes UI initialization when `DOMContentLoaded` fires in the browser.
 *
 * @return {void}
 */
function initPitchGeneratorApp() {
  if (typeof document === "undefined") {
    return;
  }
  fetchHealthAndConfig().catch((err) => {
    setBanner(`Unable to load service configuration: ${err.message}`, "warn");
  });

  const form = document.getElementById("pitch-form");
  if (form) {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const briefInput = document.getElementById("brief-input");
      const sessionInput = document.getElementById("session-id-input");
      const approvalCheckbox = document.getElementById(
        "require-approval-checkbox"
      );

      submitPitchRequest({
        brief: briefInput ? briefInput.value : "",
        session_id: sessionInput ? sessionInput.value : "default",
        require_approval: Boolean(
          approvalCheckbox && approvalCheckbox.checked
        ),
      });
    });
  }

  const approveBtn = document.getElementById("btn-approve-concept");
  if (approveBtn) {
    approveBtn.addEventListener("click", () => {
      submitApprovalDecision(true);
    });
  }

  const rejectBtn = document.getElementById("btn-reject-concept");
  if (rejectBtn) {
    rejectBtn.addEventListener("click", () => {
      submitApprovalDecision(false);
    });
  }

  const a2aBtn = document.getElementById("btn-inspect-a2a");
  if (a2aBtn) {
    a2aBtn.addEventListener("click", () => {
      inspectA2aAgentCard();
    });
  }

  // 綁定快速靈感標籤按鈕點擊事件
  const presetButtons = document.querySelectorAll(".btn-preset");
  const briefInput = document.getElementById("brief-input");
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      if (text && briefInput) {
        briefInput.value = text;
        briefInput.focus();
      }
    });
  });
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", initPitchGeneratorApp);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    createSafeElement,
    renderSafeMarkdown,
    setBanner,
    renderWorkflowTrace,
    renderPitchResult,
    fetchHealthAndConfig,
    setGenerateButtonLoading,
    submitPitchRequest,
    submitApprovalDecision,
    inspectA2aAgentCard,
    initPitchGeneratorApp,
  };
}
