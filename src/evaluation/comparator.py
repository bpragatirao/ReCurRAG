"""
Comparator — Loads RAG and RLM outputs and produces a unified comparison report.

Reads from outputs/rag/ and outputs/rlm/, computes metrics, and generates
a comprehensive comparison JSON that the frontend dashboard consumes.
"""

import os
import json
from datetime import datetime
from .metrics import compute_dataset_metrics


DATASET_CONFIG = {
    "long_docs": {
        "display_name": "Long Documents (arXiv Papers)",
        "description": "Tests long-context retrieval and summarization on academic research papers.",
        "has_ground_truth": False,
        "icon": "📄",
    },
    "semi_structured": {
        "display_name": "Semi-Structured (Wine Quality)",
        "description": "Tests tabular reasoning on structured CSV data from UCI ML Repository.",
        "has_ground_truth": False,
        "icon": "📊",
    },
    "multi_hop": {
        "display_name": "Multi-Hop QA (HotpotQA)",
        "description": "Tests multi-step reasoning requiring evidence from multiple documents.",
        "has_ground_truth": True,
        "icon": "🔗",
    },
}


def load_results(pipeline: str, dataset: str) -> dict:
    """Load results JSON for a given pipeline and dataset."""
    path = os.path.join("outputs", pipeline, dataset, f"{dataset}_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compare_all_datasets() -> dict:
    """
    Compare RAG and RLM results across all three datasets.

    Returns:
        Comprehensive comparison dict ready for JSON export.
    """
    comparison = {
        "title": "ReCurRAG — RAG vs RLM Comparison Report",
        "generated_at": datetime.now().isoformat(),
        "datasets": {},
        "overall_summary": {},
    }

    all_rag_quality = []
    all_rlm_quality = []
    all_rag_latency = []
    all_rlm_latency = []
    all_rag_em = []
    all_rlm_em = []
    all_rag_f1 = []
    all_rlm_f1 = []
    all_rlm_depth = []

    for ds_key, ds_config in DATASET_CONFIG.items():
        print(f"\n{'─'*60}")
        print(f"{ds_config['icon']} Evaluating: {ds_config['display_name']}")
        print(f"{'─'*60}")

        rag_data = load_results("rag", ds_key)
        rlm_data = load_results("rlm", ds_key)

        if not rag_data or not rlm_data:
            print(f"  ⚠️  Missing data — RAG: {'✅' if rag_data else '❌'}, "
                  f"RLM: {'✅' if rlm_data else '❌'}")
            comparison["datasets"][ds_key] = {
                "config": ds_config,
                "status": "incomplete",
                "error": "Missing RAG or RLM results"
            }
            continue

        rag_results = rag_data.get("results", [])
        rlm_results = rlm_data.get("results", [])

        print(f"  RAG: {len(rag_results)} results")
        print(f"  RLM: {len(rlm_results)} results")

        # Compute metrics
        metrics = compute_dataset_metrics(
            rag_results, rlm_results,
            has_ground_truth=ds_config["has_ground_truth"]
        )

        # Collect for overall summary
        agg = metrics["aggregate"]
        all_rag_quality.extend(
            [q["rag_quality"] for q in metrics["per_query"]]
        )
        all_rlm_quality.extend(
            [q["rlm_quality"] for q in metrics["per_query"]]
        )
        all_rag_latency.append(agg["rag"]["avg_latency_s"])
        all_rlm_latency.append(agg["rlm"]["avg_latency_s"])
        all_rlm_depth.append(agg["rlm"]["avg_reasoning_depth"])

        if "exact_match" in agg.get("rag", {}):
            all_rag_em.append(agg["rag"]["exact_match"])
            all_rlm_em.append(agg["rlm"]["exact_match"])
            all_rag_f1.append(agg["rag"]["f1_score"])
            all_rlm_f1.append(agg["rlm"]["f1_score"])

        # Print summary
        print(f"\n  📊 Results:")
        print(f"     RAG Avg Quality: {agg['rag']['avg_quality']:.2f}")
        print(f"     RLM Avg Quality: {agg['rlm']['avg_quality']:.2f}")
        print(f"     RAG Avg Latency: {agg['rag']['avg_latency_s']:.3f}s")
        print(f"     RLM Avg Latency: {agg['rlm']['avg_latency_s']:.3f}s")
        print(f"     RLM Avg Depth:   {agg['rlm']['avg_reasoning_depth']:.1f}")

        if "exact_match" in agg.get("rag", {}):
            print(f"     RAG EM: {agg['rag']['exact_match']:.2%} | "
                  f"F1: {agg['rag']['f1_score']:.4f}")
            print(f"     RLM EM: {agg['rlm']['exact_match']:.2%} | "
                  f"F1: {agg['rlm']['f1_score']:.4f}")

        comparison["datasets"][ds_key] = {
            "config": ds_config,
            "status": "complete",
            "rag_metadata": rag_data.get("metadata", {}),
            "rlm_metadata": rlm_data.get("metadata", {}),
            "metrics": metrics,
        }

    # Overall summary
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    comparison["overall_summary"] = {
        "total_datasets_evaluated": sum(
            1 for d in comparison["datasets"].values() if d.get("status") == "complete"
        ),
        "rag": {
            "avg_quality": safe_avg(all_rag_quality),
            "avg_latency_s": safe_avg(all_rag_latency),
        },
        "rlm": {
            "avg_quality": safe_avg(all_rlm_quality),
            "avg_latency_s": safe_avg(all_rlm_latency),
            "avg_reasoning_depth": safe_avg(all_rlm_depth),
        },
    }

    if all_rag_em:
        comparison["overall_summary"]["rag"]["avg_exact_match"] = safe_avg(all_rag_em)
        comparison["overall_summary"]["rag"]["avg_f1"] = safe_avg(all_rag_f1)
        comparison["overall_summary"]["rlm"]["avg_exact_match"] = safe_avg(all_rlm_em)
        comparison["overall_summary"]["rlm"]["avg_f1"] = safe_avg(all_rlm_f1)

    return comparison


def save_comparison(comparison: dict, output_path: str = "outputs/comparison_report.json"):
    """Save the comparison report as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Comparison report saved to: {output_path}")
    return output_path
