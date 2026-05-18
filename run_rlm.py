"""
RLM Runner — Executes the Recursive Language Model pipeline on all three
datasets and stores results for comparison with RAG.

Usage:
    python run_rlm.py                  # Run all datasets
    python run_rlm.py --dataset long_docs        # Run only Long-Docs
    python run_rlm.py --dataset semi_structured  # Run only Semi-Structured
    python run_rlm.py --dataset multi_hop        # Run only Multi-Hop QA

Output:
    Results are saved to outputs/rlm/<dataset_type>/<dataset_type>_results.json
    These mirror the RAG outputs for direct comparison in the evaluation stage.
"""

import os
import sys
import json
import argparse
import time

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rlm.pipeline import RLMPipeline


# ---------------------------------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------------------------------

DATASETS = {
    "long_docs": {
        "data_path": "data/raw/Long-Docs/papers/",
        "data_type": "long_docs",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_iterations": 8,
    },
    "semi_structured": {
        "data_path": "data/raw/Semi-Structured/wine+quality/",
        "data_type": "semi_structured",
        "chunk_size": 800,
        "chunk_overlap": 150,
        "max_iterations": 8,
    },
    "multi_hop": {
        "data_path": "data/raw/Multi-HopQA/hotpotqa.json",
        "data_type": "multi_hop",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "max_iterations": 10,
    },
}


def load_queries(config_path: str = "configs/queries.json") -> dict:
    """Load the query configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def run_long_docs(queries_config: dict):
    """Run RLM pipeline on arXiv Long Documents."""
    config = DATASETS["long_docs"]
    pipeline = RLMPipeline(**config)
    pipeline.ingest()

    questions = queries_config["long_docs"]["questions"]
    print(f"\n📝 Running {len(questions)} queries on Long-Docs (RLM)...\n")
    results = pipeline.run_batch(questions)
    output_path = pipeline.save_results(results)

    return results, output_path


def run_semi_structured(queries_config: dict):
    """Run RLM pipeline on Wine Quality CSVs."""
    config = DATASETS["semi_structured"]
    pipeline = RLMPipeline(**config)
    pipeline.ingest()

    questions = queries_config["semi_structured"]["questions"]
    print(f"\n📝 Running {len(questions)} queries on Semi-Structured (RLM)...\n")
    results = pipeline.run_batch(questions)
    output_path = pipeline.save_results(results)

    return results, output_path


def run_multi_hop(queries_config: dict):
    """Run RLM pipeline on HotpotQA Multi-Hop QA."""
    config = DATASETS["multi_hop"]
    pipeline = RLMPipeline(**config)
    pipeline.ingest()

    max_samples = queries_config["multi_hop"].get("max_eval_samples", 50)
    print(f"\n📝 Running HotpotQA evaluation ({max_samples} samples, RLM)...\n")
    results = pipeline.run_hotpotqa_evaluation(max_samples=max_samples)
    output_path = pipeline.save_results(results)

    return results, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run RLM (Recursive Language Model) pipeline on datasets"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "long_docs", "semi_structured", "multi_hop"],
                        help="Which dataset to process")
    args = parser.parse_args()

    # Load query configuration
    queries_config = load_queries()

    print("\n" + "=" * 70)
    print("🧠 ReCurRAG — Recursive Language Model (RLM) Pipeline Runner")
    print("=" * 70)
    print("Unlike RAG, the RLM agent will use iterative tool calls to:")
    print("  Plan → Tool Use → Reason → Refine → Aggregate")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    try:
        if args.dataset in ("all", "long_docs"):
            print("\n" + "─" * 70)
            print("📄 DATASET 1: Long Documents (arXiv Papers)")
            print("─" * 70)
            results, path = run_long_docs(queries_config)
            all_results["long_docs"] = {
                "num_results": len(results),
                "path": path,
                "avg_tool_calls": round(
                    sum(r["total_tool_calls"] for r in results) / len(results), 1
                ),
                "avg_depth": round(
                    sum(r["reasoning_depth"] for r in results) / len(results), 1
                )
            }

        if args.dataset in ("all", "semi_structured"):
            print("\n" + "─" * 70)
            print("📊 DATASET 2: Semi-Structured (Wine Quality)")
            print("─" * 70)
            results, path = run_semi_structured(queries_config)
            all_results["semi_structured"] = {
                "num_results": len(results),
                "path": path,
                "avg_tool_calls": round(
                    sum(r["total_tool_calls"] for r in results) / len(results), 1
                ),
                "avg_depth": round(
                    sum(r["reasoning_depth"] for r in results) / len(results), 1
                )
            }

        if args.dataset in ("all", "multi_hop"):
            print("\n" + "─" * 70)
            print("🔗 DATASET 3: Multi-Hop QA (HotpotQA)")
            print("─" * 70)
            results, path = run_multi_hop(queries_config)
            all_results["multi_hop"] = {
                "num_results": len(results),
                "path": path,
                "avg_tool_calls": round(
                    sum(r["total_tool_calls"] for r in results) / len(results), 1
                ),
                "avg_depth": round(
                    sum(r["reasoning_depth"] for r in results) / len(results), 1
                )
            }

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 70)
    print("📋 RLM EXECUTION SUMMARY")
    print("=" * 70)
    for ds_name, info in all_results.items():
        print(f"  {ds_name:20s} → {info['num_results']} results "
              f"(avg {info['avg_tool_calls']} tools, depth {info['avg_depth']}) "
              f"→ {info['path']}")
    print(f"\n  Total execution time: {total_time:.1f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
