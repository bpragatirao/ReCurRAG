"""
RLM Pipeline — Orchestrates the Recursive Language Model workflow.

The RLM pipeline uses the same data ingestion and embedding as RAG, but
replaces the single retrieve-then-generate step with an iterative
agent-based reasoning loop.

Workflow: Query → Plan → Tool Use → Reason → Refine → Aggregate

Supports three dataset types:
  1. 'long_docs'        — arXiv PDF papers (Long Documents)
  2. 'semi_structured'  — Wine Quality CSVs (Semi-Structured)
  3. 'multi_hop'        — HotpotQA JSON (Multi-Hop QA)

Results are saved to outputs/rlm/ in the same JSON format as RAG outputs
to enable direct comparison in the evaluation stage.
"""

import os
import json
import time
from datetime import datetime

# Reuse the RAG loader and embedder for data ingestion
from ..rag.loader import load_documents, chunk_text
from ..rag.embedder import create_embeddings
from .agent import RLMAgent


class RLMPipeline:
    """
    End-to-end RLM pipeline:
    Load → Chunk → Embed → Agent(Plan → Tool Use → Reason → Refine) → Answer
    """

    def __init__(self, data_path: str, data_type: str = "long_docs",
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 max_iterations: int = 8, model: str = "flan-t5-base",
                 output_dir: str = "outputs/rlm"):
        """
        Initialize the RLM pipeline.

        Args:
            data_path: Path to the data directory or file.
            data_type: One of 'long_docs', 'semi_structured', 'multi_hop'.
            chunk_size: Number of characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            max_iterations: Max tool-use iterations per query.
            model: Model identifier (uses local HuggingFace model).
            output_dir: Directory to save results.
        """
        self.data_path = data_path
        self.data_type = data_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_iterations = max_iterations
        self.model = model
        self.output_dir = output_dir
        self.vector_store = None
        self.documents = []
        self.chunks = []
        self.agent = None
        self.ingest_metadata = {}

    def ingest(self):
        """
        Load documents, chunk them, create FAISS embeddings, and
        initialize the RLM agent with tools.
        """
        print(f"\n{'='*60}")
        print(f"RLM Ingestion — Dataset Type: {self.data_type}")
        print(f"{'='*60}")

        # Step 1: Load documents (reuses RAG loader)
        print(f"\n[1/4] Loading documents from: {self.data_path}")
        start_time = time.time()
        self.documents = load_documents(self.data_path, data_type=self.data_type)
        load_time = time.time() - start_time
        print(f"  → Loaded {len(self.documents)} document(s) in {load_time:.2f}s")

        # Step 2: Chunk documents
        print(f"\n[2/4] Chunking documents (size={self.chunk_size}, overlap={self.chunk_overlap})")
        start_time = time.time()
        self.chunks = chunk_text(self.documents,
                                 chunk_size=self.chunk_size,
                                 overlap=self.chunk_overlap)
        chunk_time = time.time() - start_time
        print(f"  → Created {len(self.chunks)} chunks in {chunk_time:.2f}s")

        # Step 3: Create embeddings
        print(f"\n[3/4] Creating FAISS embeddings...")
        start_time = time.time()
        self.vector_store = create_embeddings(self.chunks)
        embed_time = time.time() - start_time
        print(f"  → Embeddings created in {embed_time:.2f}s")

        # Step 4: Initialize the RLM agent
        print(f"\n[4/4] Initializing RLM Agent (model={self.model}, "
              f"max_iter={self.max_iterations})...")
        self.agent = RLMAgent(
            vector_store=self.vector_store,
            documents=self.documents,
            chunks=self.chunks,
            data_type=self.data_type,
            max_iterations=self.max_iterations,
            model=self.model
        )
        print(f"  → Agent initialized with {len(self.agent.tool_executor.tool_call_log)} tools ready")

        # Store metadata
        self.ingest_metadata = {
            "data_type": self.data_type,
            "data_path": self.data_path,
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_iterations": self.max_iterations,
            "model": self.model,
            "load_time_s": round(load_time, 2),
            "chunk_time_s": round(chunk_time, 2),
            "embed_time_s": round(embed_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n✅ RLM ingestion complete!")
        return self

    def query(self, question: str) -> dict:
        """
        Run a query through the RLM agent's recursive reasoning loop.

        Args:
            question: The question to answer.

        Returns:
            Dict with: question, answer, reasoning_trace, tool_calls,
                       num_iterations, reasoning_depth, latency_s.
        """
        if self.agent is None:
            raise ValueError("Run ingest() first before querying!")

        start_time = time.time()
        result = self.agent.query(question)
        latency = time.time() - start_time

        result["latency_s"] = round(latency, 3)
        return result

    def run_batch(self, questions: list) -> list:
        """
        Run a batch of queries through the RLM agent.

        Args:
            questions: List of question strings.

        Returns:
            List of result dicts.
        """
        results = []
        for i, q in enumerate(questions):
            print(f"  Query {i+1}/{len(questions)}: {q[:80]}...")
            result = self.query(q)
            print(f"    → {result['num_iterations']} iterations, "
                  f"{result['total_tool_calls']} tool calls, "
                  f"{result['latency_s']}s")
            results.append(result)
        return results

    def run_hotpotqa_evaluation(self, max_samples: int = 50) -> list:
        """
        Run RLM on HotpotQA samples using their built-in questions.

        Args:
            max_samples: Number of samples to evaluate.

        Returns:
            List of result dicts with ground-truth included.
        """
        if self.data_type != "multi_hop":
            raise ValueError("This method is only for 'multi_hop' data type.")

        results = []
        samples = self.documents[:max_samples]

        for i, doc in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] {doc['question'][:80]}...")

            result = self.query(doc["question"])

            # Attach ground truth for evaluation
            result["ground_truth_answer"] = doc.get("answer", "")
            result["supporting_facts"] = doc.get("supporting_facts", [])
            result["level"] = doc.get("level", "")
            result["type"] = doc.get("type", "")

            print(f"    → {result['num_iterations']} iterations, "
                  f"{result['total_tool_calls']} tool calls, "
                  f"depth={result['reasoning_depth']}")
            results.append(result)

        return results

    def save_results(self, results: list, filename: str = None):
        """
        Save results to the outputs directory as JSON.

        Args:
            results: List of result dicts.
            filename: Custom filename. Defaults to '{data_type}_results.json'.
        """
        dataset_output_dir = os.path.join(self.output_dir, self.data_type)
        os.makedirs(dataset_output_dir, exist_ok=True)

        if filename is None:
            filename = f"{self.data_type}_results.json"

        output_path = os.path.join(dataset_output_dir, filename)

        # Compute aggregate metrics
        avg_latency = (
            sum(r["latency_s"] for r in results) / len(results)
        ) if results else 0

        avg_iterations = (
            sum(r["num_iterations"] for r in results) / len(results)
        ) if results else 0

        avg_tool_calls = (
            sum(r["total_tool_calls"] for r in results) / len(results)
        ) if results else 0

        avg_reasoning_depth = (
            sum(r["reasoning_depth"] for r in results) / len(results)
        ) if results else 0

        output_data = {
            "pipeline": "rlm",
            "metadata": self.ingest_metadata,
            "results": results,
            "summary": {
                "total_queries": len(results),
                "avg_latency_s": round(avg_latency, 3),
                "avg_iterations": round(avg_iterations, 2),
                "avg_tool_calls": round(avg_tool_calls, 2),
                "avg_reasoning_depth": round(avg_reasoning_depth, 2),
                "timestamp": datetime.now().isoformat(),
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_path}")
        return output_path
