"""
RAG Runner — Executes the RAG pipeline on all three datasets and stores results.

Usage:
    python run_rag.py                  # Run all datasets
    python run_rag.py --dataset long_docs        # Run only Long-Docs
    python run_rag.py --dataset semi_structured  # Run only Semi-Structured
    python run_rag.py --dataset multi_hop        # Run only Multi-Hop QA

Output:
    Results are saved to outputs/rag/<dataset_type>/<dataset_type>_results.json
    These can later be compared against RLM outputs in the evaluation stage.
"""

import os
import sys
import json
import argparse
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------------------------------

DATASETS = {
    "long_docs": {
        "data_path": "data/raw/Long-Docs/papers/",
        "data_type": "long_docs",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
    },
    "semi_structured": {
        "data_path": "data/raw/Semi-Structured/wine+quality/",
        "data_type": "semi_structured",
        "chunk_size": 800,
        "chunk_overlap": 150,
        "top_k": 5,
    },
    "multi_hop": {
        "data_path": "data/raw/Multi-HopQA/hotpotqa.json",
        "data_type": "multi_hop",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
    },
}


def load_queries(config_path: str = "configs/queries.json") -> dict:
    """Load the query configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def run_long_docs(queries_config: dict):
    """Run RAG pipeline on arXiv Long Documents."""
    config = DATASETS["long_docs"]
    pipeline = RAGPipeline(**config)
    pipeline.ingest()

    questions = queries_config["long_docs"]["questions"]
    print(f"\n📝 Running {len(questions)} queries on Long-Docs...\n")
    results = pipeline.run_batch(questions)
    output_path = pipeline.save_results(results)

    return results, output_path


def run_semi_structured(queries_config: dict):
    """Run RAG pipeline on Wine Quality CSVs."""
    config = DATASETS["semi_structured"]
    pipeline = RAGPipeline(**config)
    pipeline.ingest()

    questions = queries_config["semi_structured"]["questions"]
    print(f"\n📝 Running {len(questions)} queries on Semi-Structured...\n")
    results = pipeline.run_batch(questions)
    output_path = pipeline.save_results(results)

    return results, output_path


def run_multi_hop(queries_config: dict):
    """Run RAG pipeline on HotpotQA Multi-Hop QA."""
    config = DATASETS["multi_hop"]
    pipeline = RAGPipeline(**config)
    pipeline.ingest()

    max_samples = queries_config["multi_hop"].get("max_eval_samples", 50)
    print(f"\n📝 Running HotpotQA evaluation ({max_samples} samples)...\n")
    results = pipeline.run_hotpotqa_evaluation(max_samples=max_samples)
    output_path = pipeline.save_results(results)

    return results, output_path


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline on datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "long_docs", "semi_structured", "multi_hop"],
                        help="Which dataset to process")
    args = parser.parse_args()

    # Load query configuration
    queries_config = load_queries()

    print("\n" + "=" * 70)
    print("🚀 ReCurRAG — Standard RAG Pipeline Runner")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    try:
        if args.dataset in ("all", "long_docs"):
            print("\n" + "─" * 70)
            print("📄 DATASET 1: Long Documents (arXiv Papers)")
            print("─" * 70)
            results, path = run_long_docs(queries_config)
            all_results["long_docs"] = {"num_results": len(results), "path": path}

        if args.dataset in ("all", "semi_structured"):
            print("\n" + "─" * 70)
            print("📊 DATASET 2: Semi-Structured (Wine Quality)")
            print("─" * 70)
            results, path = run_semi_structured(queries_config)
            all_results["semi_structured"] = {"num_results": len(results), "path": path}

        if args.dataset in ("all", "multi_hop"):
            print("\n" + "─" * 70)
            print("🔗 DATASET 3: Multi-Hop QA (HotpotQA)")
            print("─" * 70)
            results, path = run_multi_hop(queries_config)
            all_results["multi_hop"] = {"num_results": len(results), "path": path}

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 70)
    print("📋 EXECUTION SUMMARY")
    print("=" * 70)
    for ds_name, info in all_results.items():
        print(f"  {ds_name:20s} → {info['num_results']} results → {info['path']}")
    print(f"\n  Total execution time: {total_time:.1f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
