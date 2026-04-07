from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric
from metric_factory.base import GEVAL_THRESHOLD, JUDGE_MODEL


def response_quality_metric(judge_model: str | None = None) -> GEval:
    model = judge_model or JUDGE_MODEL
    return GEval(
        name="ResponseQuality",
        criteria=(
            "You are evaluating an agricultural assistant's response quality — "
            "how completely it answered the query and whether it properly cited its source.\n\n"

            "## STEP 0 — Classify the response\n"
            "Classify the ACTUAL OUTPUT as one of:\n"
            "  FULL    — Substantive answer with most or all key facts.\n"
            "  PARTIAL — Answered but key facts are missing or vague.\n"
            "  DEFLECT — No useful information; only a redirect or clarification question.\n"
            "  EMPTY   — Blank, error message, or non-answer.\n\n"

            "## STEP 1 — Score Dimension A: Information Completeness (0–1)\n"
            "Use the checklist for the query type:\n"
            "  Mandi price   → commodity + price + unit (₹/quintal) + market name + recency signal.\n"
            "  Weather       → location + temperature (with unit) + conditions + advisory if relevant.\n"
            "  Scheme status → scheme name + eligibility/status + next actionable step.\n"
            "  Pest/crop     → problem identified + treatment + dose/method + timing.\n"
            "  General agri  → direct factual answer + critical caveats (region/season).\n\n"
            "Scores:\n"
            "  1.0 — All checklist items present and correct.\n"
            "  0.7 — One minor item missing (e.g. price present but no market name).\n"
            "  0.4 — One major item missing (e.g. treatment given but no dose).\n"
            "  0.0 — No substantive information; response is DEFLECT or EMPTY.\n"
            "Padding rule: if ≥3 sentences are unrelated or repetitive, deduct 0.1 (floor 0.0).\n"
            "DEFLECT/EMPTY cap: cannot score above 0.0 on dimension A.\n\n"

            "## STEP 2 — Score Dimension B: Source Citation (0–1)\n"
            "  1.0 — Bold citation on its own dedicated line in any supported language.\n"
            "        e.g. '**Source: Mandi Prices**' or '**स्रोत: मंडी भाव**'\n"
            "  0.5 — Source mentioned but not bolded OR not on its own line.\n"
            "  0.0 — No source mentioned.\n\n"
            "Exempt (mark B = N/A, exclude from final score) if:\n"
            "  - Response is DEFLECT or EMPTY.\n"
            "  - Query is purely procedural with no external data (e.g. 'how do I register for PM Kisan?').\n"
            "Never penalise for non-English citation text.\n\n"

            "## Final Score\n"
            "  B scored → Final = A×0.65 + B×0.35\n"
            "  B = N/A  → Final = A×1.00\n"
            "Round to two decimal places."
        ),
        evaluation_steps=[
            "Classify the ACTUAL OUTPUT as FULL, PARTIAL, DEFLECT, or EMPTY. "
            "In one sentence, state the classification and the main reason.",

            "Identify the query type (Mandi price / Weather / Scheme status / Pest/crop / General agri). "
            "List the required checklist items, then mark each as: present, missing, or partial.",

            "Score dimension A using anchors {0.0, 0.4, 0.7, 1.0}. "
            "Apply the padding deduction if ≥3 unrelated/repetitive sentences exist. "
            "Apply the DEFLECT/EMPTY cap (0.0) if applicable. State the final A score.",

            "Evaluate dimension B. Check if the citation exemption applies — if so, mark B = N/A and state why. "
            "Otherwise locate the citation: is it bold? is it on its own line? "
            "Assign B ∈ {0.0, 0.5, 1.0} and quote the citation as evidence.",

            "Compute the final score using the correct formula. "
            "State A, B (or N/A), and the final score rounded to two decimal places.",

            "Write a two-sentence verdict: (1) response type, final score, and primary driver; "
            "(2) what prevented a complete response, if anything.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        rubric=[
            Rubric(
                score_range=(0, 1),
                expected_outcome=(
                    "DEFLECT or EMPTY — no useful information returned. Final score ≤ 0.10."
                ),
            ),
            Rubric(
                score_range=(2, 4),
                expected_outcome=(
                    "PARTIAL with a major failure — critical checklist item missing, "
                    "or source citation entirely absent when required."
                ),
            ),
            Rubric(
                score_range=(5, 7),
                expected_outcome=(
                    "PARTIAL with a minor gap — one non-critical item missing, "
                    "or citation exists but fails one formatting requirement."
                ),
            ),
            Rubric(
                score_range=(8, 10),
                expected_outcome=(
                    "FULL — all checklist items present, bold citation on its own line "
                    "(or correctly exempted for DEFLECT/EMPTY/procedural responses)."
                ),
            ),
        ],
        threshold=GEVAL_THRESHOLD,
        model=model,
        async_mode=True,
        verbose_mode=True,
    )