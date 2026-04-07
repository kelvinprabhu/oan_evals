from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

from metric_factory.base import GEVAL_THRESHOLD, JUDGE_MODEL


def response_validity_metric(judge_model: str | None = None) -> GEval:
    model = judge_model or JUDGE_MODEL
    return GEval(
        name="ResponseValidity",
        criteria=(
            "You are evaluating an agricultural assistant's response to a user query.\n\n"

            "## STEP 0 — Classify the query\n"
            "Before scoring, classify the INPUT into one of two tracks:\n"
            "  TRACK-V (Valid): The query is about ANY of the following topics — "
            "farming, crops, seeds, soil health, irrigation, pest control, crop diseases, "
            "weather forecasts (weather is a core agricultural topic — ALWAYS TRACK-V), "
            "mandi/market prices (ALWAYS TRACK-V), livestock, fisheries, "
            "government schemes (PM Kisan, PMFBY, SHC, KCC, PMKSY, SATHI, PMASHA, AIF, SMAM, PDMC, etc.), "
            "fertilizers, storage, or any agriculture-adjacent topic. "
            "When in doubt, default to TRACK-V — this assistant serves farmers broadly.\n"
            "  TRACK-D (Decline): The query is clearly and unambiguously out-of-scope — "
            "topics with zero relevance to agriculture or farming, such as politics, "
            "entertainment, general coding, or personal advice unrelated to farming. "
            "Do NOT classify weather, mandi prices, soil, livestock, or scheme queries as TRACK-D.\n\n"
            "Apply the rubric for the detected track. A response that answers an "
            "out-of-scope query instead of declining scores 0.0 on the TRACK-D rubric.\n\n"

            "---\n\n"
        ),
        evaluation_steps=[
            # Step 0 — classify
            "Read the INPUT carefully. Classify it as TRACK-V or TRACK-D. "
            "Remember: weather forecasts, mandi prices, soil health, livestock, fisheries, "
            "and government scheme queries are ALWAYS TRACK-V — never classify these as TRACK-D. "
            "Only classify as TRACK-D if the query is clearly unrelated to agriculture "
            "(e.g. politics, entertainment, general coding). "
            "State your classification and the single main reason for it before proceeding.",

            # TRACK-V steps
            "IF TRACK-V — Score dimension A (Direct Answer): Does the response open "
            "immediately with the answer, or is there a preamble? Is the information "
            "actionable? Apply the PMFBY special rule if relevant. Assign A ∈ {0.0, 0.4, 0.7, 1.0} "
            "and quote the opening sentence as evidence.",

            "IF TRACK-V — Score dimension B (Answer Relevance): Read every sentence in "
            "the ACTUAL OUTPUT. Does each sentence contribute directly to answering the "
            "query? Flag any tangential, repetitive, or unsolicited sentences by quoting "
            "them. Assign B ∈ {0.0, 0.4, 0.7, 1.0}.",

            "IF TRACK-V — Score dimension C (Completeness + Follow-up): Identify what "
            "details are necessary to act on this answer (doses, dates, amounts, steps, "
            "eligibility). Are they all present? Is there exactly one relevant follow-up "
            "question at the end? Assign C ∈ {0.0, 0.4, 0.7, 1.0}.",

            "IF TRACK-V — Compute final: Final = A×0.40 + B×0.35 + C×0.25. "
            "Round to two decimal places. State A, B, C, and the final score.",

            # TRACK-D steps
            "IF TRACK-D — Score dimension X (Scope Adherence): Did the response refuse "
            "to provide the out-of-scope content? Did it accidentally answer any part of "
            "the query? Quote the key sentence that shows the decline or the slip. "
            "Assign X ∈ {0.0, 0.5, 1.0}.",

            "IF TRACK-D — Score dimension Y (Polite Redirection): Is the tone warm and "
            "non-dismissive? Does it offer a concrete agricultural topic or question the "
            "user could pivot to? Assign Y ∈ {0.0, 0.5, 1.0}.",

            "IF TRACK-D — Compute final: Final = X×0.70 + Y×0.30. "
            "Round to two decimal places. State X, Y, and the final score.",

            # Shared verdict step
            "Write a one-sentence verdict: state the track, the final score, and the "
            "single biggest factor that determined it.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        rubric=[
            Rubric(
                score_range=(0, 2),
                expected_outcome=(
                    "TRACK-V: Response fails entirely — no answer, wrong topic, or unexplained refusal. "
                    "TRACK-D: Response answers the out-of-scope query instead of declining."
                ),
            ),
            Rubric(
                score_range=(3, 5),
                expected_outcome=(
                    "TRACK-V: Answer is indirect or buried behind preamble; noticeable padding or topic drift; "
                    "key details (doses, dates, steps) are missing; follow-up is absent or irrelevant. "
                    "TRACK-D: Response partially or rudely declines with no path forward for the user."
                ),
            ),
            Rubric(
                score_range=(6, 8),
                expected_outcome=(
                    "TRACK-V: Answer is present and mostly relevant with minor filler or one missing key detail; "
                    "follow-up question is generic or slightly off-topic. "
                    "TRACK-D: Polite decline but redirection to agriculture is vague or absent."
                ),
            ),
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "TRACK-V: Immediate, actionable answer with no preamble; every sentence is relevant; "
                    "all key details present; ends with exactly one specific, on-topic follow-up question. "
                    "TRACK-D: Warm, non-dismissive decline with a concrete, useful agricultural redirect."
                ),
            ),
        ],
        threshold=GEVAL_THRESHOLD,
        model=model,
        async_mode=True,
        verbose_mode=True,
    )