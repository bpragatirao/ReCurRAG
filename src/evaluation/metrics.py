"""
Evaluation Metrics — Computes comparison metrics between RAG and RLM outputs.

Metrics:
  - Exact Match (EM): Whether the predicted answer contains the ground truth
  - F1 Score: Token-level overlap between prediction and ground truth
  - Reasoning Depth: Number of tool calls / reasoning steps (RLM-specific)
  - Context Coverage: How much of the relevant context was captured
  - Latency: Response time comparison
"""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/whitespace for fair comparison."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Check if the ground truth answer appears in the prediction.
    Returns 1.0 if match, 0.0 otherwise.
    """
    if not ground_truth or not prediction:
        return 0.0
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    # Check containment (more forgiving than exact string match)
    return 1.0 if gt_norm in pred_norm else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.
    """
    if not ground_truth or not prediction:
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def answer_quality_score(answer: str) -> float:
    """
    Heuristic score for answer quality (0-1).
    Checks for error messages, length, and content indicators.
    """
    if not answer:
        return 0.0

    # Check for error indicators
    error_patterns = [
        "error generating response",
        "error calling llm",
        "error code: 429",
        "exceeded your current quota",
        "i don't know",
        "no answer generated",
        "insufficient_quota"
    ]
    answer_lower = answer.lower()
    for pattern in error_patterns:
        if pattern in answer_lower:
            return 0.0

    # Score based on answer substance
    word_count = len(answer.split())
    if word_count < 3:
        return 0.2
    elif word_count < 10:
        return 0.4
    elif word_count < 30:
        return 0.6
    elif word_count < 100:
        return 0.8
    else:
        return 1.0


def compute_dataset_metrics(rag_results: list, rlm_results: list,
                             has_ground_truth: bool = False) -> dict:
    """
    Compute comparison metrics between RAG and RLM results for a dataset.

    Args:
        rag_results: List of RAG result dicts.
        rlm_results: List of RLM result dicts.
        has_ground_truth: Whether results include ground_truth_answer.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    per_query = []
    rag_metrics = {"latencies": [], "em_scores": [], "f1_scores": [],
                   "quality_scores": []}
    rlm_metrics = {"latencies": [], "em_scores": [], "f1_scores": [],
                   "quality_scores": [], "tool_calls": [],
                   "reasoning_depths": [], "iterations": []}

    num_queries = min(len(rag_results), len(rlm_results))

    for i in range(num_queries):
        rag_r = rag_results[i]
        rlm_r = rlm_results[i]

        query_data = {
            "query_id": i,
            "question": rag_r.get("question", rlm_r.get("question", "")),
            "rag_answer": rag_r.get("answer", ""),
            "rlm_answer": rlm_r.get("answer", ""),
            "rag_latency_s": rag_r.get("latency_s", 0),
            "rlm_latency_s": rlm_r.get("latency_s", 0),
            "rlm_tool_calls": rlm_r.get("total_tool_calls", 0),
            "rlm_reasoning_depth": rlm_r.get("reasoning_depth", 0),
            "rlm_iterations": rlm_r.get("num_iterations", 0),
        }

        # Answer quality
        rag_quality = answer_quality_score(rag_r.get("answer", ""))
        rlm_quality = answer_quality_score(rlm_r.get("answer", ""))
        query_data["rag_quality"] = rag_quality
        query_data["rlm_quality"] = rlm_quality

        rag_metrics["quality_scores"].append(rag_quality)
        rlm_metrics["quality_scores"].append(rlm_quality)
        rag_metrics["latencies"].append(rag_r.get("latency_s", 0))
        rlm_metrics["latencies"].append(rlm_r.get("latency_s", 0))
        rlm_metrics["tool_calls"].append(rlm_r.get("total_tool_calls", 0))
        rlm_metrics["reasoning_depths"].append(rlm_r.get("reasoning_depth", 0))
        rlm_metrics["iterations"].append(rlm_r.get("num_iterations", 0))

        # Ground truth metrics (only for multi-hop)
        if has_ground_truth:
            gt = rag_r.get("ground_truth_answer", "")
            if gt:
                rag_em = exact_match(rag_r.get("answer", ""), gt)
                rlm_em = exact_match(rlm_r.get("answer", ""), gt)
                rag_f1 = f1_score(rag_r.get("answer", ""), gt)
                rlm_f1 = f1_score(rlm_r.get("answer", ""), gt)

                query_data["ground_truth"] = gt
                query_data["rag_em"] = rag_em
                query_data["rlm_em"] = rlm_em
                query_data["rag_f1"] = rag_f1
                query_data["rlm_f1"] = rlm_f1
                query_data["level"] = rag_r.get("level", "")

                rag_metrics["em_scores"].append(rag_em)
                rlm_metrics["em_scores"].append(rlm_em)
                rag_metrics["f1_scores"].append(rag_f1)
                rlm_metrics["f1_scores"].append(rlm_f1)

        per_query.append(query_data)

    # Compute aggregates
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    aggregate = {
        "num_queries": num_queries,
        "rag": {
            "avg_latency_s": safe_avg(rag_metrics["latencies"]),
            "avg_quality": safe_avg(rag_metrics["quality_scores"]),
        },
        "rlm": {
            "avg_latency_s": safe_avg(rlm_metrics["latencies"]),
            "avg_quality": safe_avg(rlm_metrics["quality_scores"]),
            "avg_tool_calls": safe_avg(rlm_metrics["tool_calls"]),
            "avg_reasoning_depth": safe_avg(rlm_metrics["reasoning_depths"]),
            "avg_iterations": safe_avg(rlm_metrics["iterations"]),
        },
    }

    if has_ground_truth and rag_metrics["em_scores"]:
        aggregate["rag"]["exact_match"] = safe_avg(rag_metrics["em_scores"])
        aggregate["rag"]["f1_score"] = safe_avg(rag_metrics["f1_scores"])
        aggregate["rlm"]["exact_match"] = safe_avg(rlm_metrics["em_scores"])
        aggregate["rlm"]["f1_score"] = safe_avg(rlm_metrics["f1_scores"])

    return {
        "per_query": per_query,
        "aggregate": aggregate,
    }
