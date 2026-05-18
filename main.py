"""
main.py — ReCurRAG Evaluation & Dashboard Launcher

Compares RAG vs RLM results across all three datasets, generates a
comparison report, and launches a local web dashboard to visualize
the differences.

Usage:
    python main.py                    # Run evaluation + launch dashboard
    python main.py --eval-only        # Run evaluation only (no dashboard)
    python main.py --dashboard-only   # Launch dashboard only (skip eval)
    python main.py --port 8080        # Custom port for dashboard
"""

import os
import sys
import json
import argparse
import http.server
import socketserver
import webbrowser
import threading

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.comparator import compare_all_datasets, save_comparison


COMPARISON_REPORT_PATH = "outputs/comparison_report.json"
DASHBOARD_DIR = "frontend"
DEFAULT_PORT = 8000


def run_evaluation():
    """Run the full RAG vs RLM comparison and save the report."""
    print("\n" + "=" * 70)
    print("📊 ReCurRAG — RAG vs RLM Evaluation")
    print("=" * 70)

    comparison = compare_all_datasets()
    report_path = save_comparison(comparison, COMPARISON_REPORT_PATH)

    # Print overall summary
    summary = comparison.get("overall_summary", {})
    print("\n" + "=" * 70)
    print("📋 OVERALL COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n  Datasets evaluated: {summary.get('total_datasets_evaluated', 0)}")

    rag_s = summary.get("rag", {})
    rlm_s = summary.get("rlm", {})

    print(f"\n  {'Metric':<30} {'RAG':>12} {'RLM':>12} {'Winner':>10}")
    print(f"  {'─'*64}")

    # Quality
    rq = rag_s.get("avg_quality", 0)
    lq = rlm_s.get("avg_quality", 0)
    winner = "RLM ✅" if lq > rq else ("RAG ✅" if rq > lq else "Tie")
    print(f"  {'Avg Answer Quality':<30} {rq:>12.2f} {lq:>12.2f} {winner:>10}")

    # Latency
    rl = rag_s.get("avg_latency_s", 0)
    ll = rlm_s.get("avg_latency_s", 0)
    winner = "RAG ✅" if rl < ll else ("RLM ✅" if ll < rl else "Tie")
    print(f"  {'Avg Latency (s)':<30} {rl:>12.3f} {ll:>12.3f} {winner:>10}")

    # Reasoning depth
    rd = rlm_s.get("avg_reasoning_depth", 0)
    print(f"  {'RLM Reasoning Depth':<30} {'N/A':>12} {rd:>12.1f} {'—':>10}")

    # EM and F1 if available
    if "avg_exact_match" in rag_s:
        rem = rag_s.get("avg_exact_match", 0)
        lem = rlm_s.get("avg_exact_match", 0)
        winner = "RLM ✅" if lem > rem else ("RAG ✅" if rem > lem else "Tie")
        print(f"  {'Exact Match (Multi-Hop)':<30} {rem:>11.1%} {lem:>11.1%} {winner:>10}")

        rf1 = rag_s.get("avg_f1", 0)
        lf1 = rlm_s.get("avg_f1", 0)
        winner = "RLM ✅" if lf1 > rf1 else ("RAG ✅" if rf1 > lf1 else "Tie")
        print(f"  {'F1 Score (Multi-Hop)':<30} {rf1:>12.4f} {lf1:>12.4f} {winner:>10}")

    print("\n" + "=" * 70)
    return report_path


def launch_dashboard(port: int = DEFAULT_PORT):
    """Launch a local HTTP server serving the frontend dashboard."""
    # Copy comparison report to frontend directory for access
    frontend_data_path = os.path.join(DASHBOARD_DIR, "comparison_report.json")
    if os.path.exists(COMPARISON_REPORT_PATH):
        import shutil
        shutil.copy2(COMPARISON_REPORT_PATH, frontend_data_path)
        print(f"  📄 Copied report to {frontend_data_path}")

    # Start HTTP server
    os.chdir(DASHBOARD_DIR)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *args: None  # Suppress logs

    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n🌐 Dashboard running at: {url}")
        print(f"   Press Ctrl+C to stop.\n")

        # Open browser
        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard stopped.")
            httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="ReCurRAG — RAG vs RLM Evaluation & Dashboard"
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only, skip dashboard")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Launch dashboard only, skip evaluation")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port for the dashboard server (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    if not args.dashboard_only:
        run_evaluation()

    if not args.eval_only:
        if not os.path.exists(COMPARISON_REPORT_PATH):
            print("❌ No comparison report found. Run evaluation first.")
            return 1

        launch_dashboard(args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
