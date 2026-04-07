from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

from metric_factory.base import GEVAL_THRESHOLD, JUDGE_MODEL, LANGUAGE_LABELS


def language_quality_metric(lang_code: str, judge_model: str | None = None) -> GEval:
    lang_label = LANGUAGE_LABELS.get(lang_code, lang_code)
    model = judge_model or JUDGE_MODEL
    return GEval(
        name=f"LanguageQuality_{lang_code}",
        criteria=(
            f"The user submitted their query in {lang_label}. "
            f"Evaluate the actual output across four dimensions:\n\n"

            f"A. LANGUAGE ADHERENCE (0–1): Is the response written in {lang_label}?\n"
            f"   1.0 — Entire response is in {lang_label}. Numbers, proper nouns (place/scheme "
            f"names), URLs, and English technical terms (PM Kisan, PMFBY, IMD, DAP, NPK) are exempt.\n"
            f"   0.5 — Mostly {lang_label} but contains full sentences or paragraphs in another language.\n"
            f"   0.0 — Response is predominantly in the wrong language.\n\n"

            f"B. GRAMMATICAL CORRECTNESS (0–1): Is the grammar correct for {lang_label}?\n"
            f"   Assess verb forms, postpositions/prepositions, and gender agreements "
            f"(where applicable). Penalise errors that impede understanding.\n\n"

            f"C. NATURAL PHRASING (0–1): Does the text read as natural, conversational "
            f"{lang_label} a farmer would easily understand?\n"
            f"   Penalise: machine-translated feel, awkward literal translations of English "
            f"idioms, unnatural word order, inappropriate slang, or overly formal bureaucratic tone.\n\n"

            "D. STRUCTURAL COHERENCE (0–1): Is the response well-organised?\n"
            "   Expected order: direct answer → source citation on its own line → "
            "follow-up question last. Bullet points only for eligibility/requirement lists. "
            "No repetition.\n\n"

            "Final score = A×0.30 + B×0.25 + C×0.30 + D×0.15\n\n"
            "A score of 0 on dimension A (wrong language entirely) MUST result in a "
            "final score ≤ 0.15, regardless of quality in other dimensions."
        ),
        evaluation_steps=[
            # Step 1 — establish the target language
            f"Identify the language of the INPUT. Confirm it is {lang_label}. "
            f"If the input is not {lang_label}, flag this and proceed using {lang_label} "
            "as the expected response language.",

            # Step 2 — score dimension A
            f"Read the ACTUAL OUTPUT and determine what fraction of the substantive text "
            f"is written in {lang_label}. Exempt: numbers, proper nouns, place names, "
            "scheme names (e.g. PM Kisan, PMFBY), URLs, and standard agricultural/technical "
            "abbreviations (IMD, DAP, NPK). "
            f"Assign A: 1.0 if wholly {lang_label}, 0.5 if significant non-{lang_label} "
            "sentences appear, 0.0 if predominantly another language.",

            # Step 3 — score dimension B
            f"Evaluate grammatical correctness for {lang_label}: check verb conjugations, "
            "postpositions/prepositions, and gender/number agreement where the language "
            "requires it. Note specific errors if any. Assign B ∈ [0, 1].",

            # Step 4 — score dimension C
            f"Evaluate naturalness: would a {lang_label}-speaking farmer find this easy "
            "to understand and natural to read? Flag machine-translated phrasing, literal "
            "English calques, unnatural word order, or register mismatches. Assign C ∈ [0, 1].",

            # Step 5 — score dimension D
            "Evaluate structural coherence: does the response lead with the direct answer, "
            "follow with source citations on their own line, and end with a follow-up "
            "question (if present)? Are bullet points limited to eligibility/requirement "
            "lists only? Is there unnecessary repetition? Assign D ∈ [0, 1].",

            # Step 6 — apply veto rule and compute final score
            "Apply the veto rule: if A = 0.0, cap the final score at 0.15. "
            "Otherwise compute: Final = A×0.30 + B×0.25 + C×0.30 + D×0.15. "
            "Round to two decimal places.",

            # Step 7 — write the verdict
            "State the scores for A, B, C, D, the veto rule outcome (applied / not applied), "
            "and the final weighted score. Cite one concrete example from the output to "
            "justify each sub-score.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        rubric=[
            Rubric(
                score_range=(0, 2),
                expected_outcome=(
                    f"Response is predominantly in the wrong language (not {lang_label}), "
                    "with severe grammar errors or entirely unintelligible phrasing. "
                    "Veto rule applies: final score is capped at 0.15."
                ),
            ),
            Rubric(
                score_range=(3, 5),
                expected_outcome=(
                    f"Response mixes {lang_label} with significant non-{lang_label} sentences, "
                    "contains grammar errors that impede understanding, "
                    "reads as machine-translated or awkwardly literal, "
                    "or has poor structural organisation (e.g. answer buried, no citation line)."
                ),
            ),
            Rubric(
                score_range=(6, 8),
                expected_outcome=(
                    f"Response is mostly in {lang_label} with only minor language slips, "
                    "grammar is largely correct but with occasional errors, "
                    "phrasing is understandable though slightly unnatural, "
                    "and structure is reasonable but not fully consistent with the expected order."
                ),
            ),
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    f"Response is entirely in {lang_label} (exemptions applied correctly), "
                    "grammar is accurate with correct verb forms, postpositions, and gender agreement, "
                    f"phrasing sounds natural to a {lang_label}-speaking farmer, "
                    "and structure follows the expected order: answer → citation on its own line → follow-up question last."
                ),
            ),
        ],
        threshold=GEVAL_THRESHOLD,
        model=model,
        async_mode=True,
        verbose_mode=True,
    )