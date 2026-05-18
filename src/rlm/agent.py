"""
RLM Agent — Recursive Language Model agent with iterative multi-step reasoning.

Uses a SCRIPTED tool-calling strategy with a local model (google/flan-t5-base).
No API keys or billing required.

Unlike RAG (single retrieve → generate), the RLM agent:
  1. Surveys available documents (get_document_summary)
  2. Searches with the original query (search_knowledge_base)
  3. Decomposes complex questions into sub-questions
  4. Searches for each sub-question separately
  5. Performs data analysis when appropriate (analyze_data)
  6. Synthesizes all evidence into a final answer
"""

from .tools import ToolExecutor
from ..utils.local_llm import (
    answer_question,
    decompose_question,
    synthesize_answer,
)


class RLMAgent:
    """
    Recursive Language Model agent that iteratively uses tools to answer
    questions through multi-step reasoning.

    Uses a scripted strategy (not LLM-driven tool selection) to ensure
    reliable, deterministic multi-hop reasoning with the local model.
    """

    def __init__(self, vector_store, documents: list, chunks: list,
                 data_type: str = "long_docs", max_iterations: int = 8,
                 model: str = "flan-t5-base"):
        """
        Initialize the RLM agent.

        Args:
            vector_store: FAISS vector store with embedded chunks.
            documents: List of loaded document dicts.
            chunks: List of text chunks.
            data_type: One of 'long_docs', 'semi_structured', 'multi_hop'.
            max_iterations: Maximum number of tool-use iterations.
            model: Model identifier (informational; always uses local model).
        """
        self.data_type = data_type
        self.max_iterations = max_iterations
        self.model = model

        # Initialize tool executor
        self.tool_executor = ToolExecutor(
            vector_store=vector_store,
            documents=documents,
            chunks=chunks,
            data_type=data_type
        )

    def query(self, question: str) -> dict:
        """
        Run the multi-step agent to answer a question.

        Strategy varies by data type:
          - long_docs:       survey → search → decompose → search subs → synthesize
          - semi_structured: survey → analyze → search → analyze deeper → synthesize
          - multi_hop:       decompose → search each sub → reason → synthesize

        Args:
            question: The question to answer.

        Returns:
            Dict with: answer, reasoning_trace, tool_calls, num_iterations,
                       reasoning_depth, etc.
        """
        # Reset tool call log for this query
        self.tool_executor.tool_call_log = []

        if self.data_type == "semi_structured":
            return self._run_semi_structured(question)
        elif self.data_type == "multi_hop":
            return self._run_multi_hop(question)
        else:
            return self._run_long_docs(question)

    # ------------------------------------------------------------------
    # Strategy: Long Documents (arXiv Papers)
    # ------------------------------------------------------------------
    def _run_long_docs(self, question: str) -> dict:
        """Multi-step reasoning for long document queries."""
        reasoning_trace = []
        evidence_pieces = []

        # Step 1: Get document overview
        summary = self.tool_executor.execute("get_document_summary", {})
        reasoning_trace.append({
            "iteration": 1, "type": "tool_call",
            "tool": "get_document_summary", "arguments": {}
        })
        reasoning_trace.append({
            "iteration": 1, "type": "tool_result",
            "tool": "get_document_summary", "result": summary[:300]
        })

        # Step 2: Broad search with original question
        search_result = self.tool_executor.execute(
            "search_knowledge_base", {"query": question, "num_results": 5}
        )
        reasoning_trace.append({
            "iteration": 2, "type": "tool_call",
            "tool": "search_knowledge_base",
            "arguments": {"query": question}
        })
        reasoning_trace.append({
            "iteration": 2, "type": "tool_result",
            "tool": "search_knowledge_base",
            "result": search_result[:300]
        })
        evidence_pieces.append((question, search_result))

        # Step 3: Decompose into sub-questions and search each
        sub_questions = decompose_question(question)
        for i, sub_q in enumerate(sub_questions):
            sub_result = self.tool_executor.execute(
                "search_knowledge_base", {"query": sub_q, "num_results": 3}
            )
            reasoning_trace.append({
                "iteration": 3 + i, "type": "tool_call",
                "tool": "search_knowledge_base",
                "arguments": {"query": sub_q}
            })
            reasoning_trace.append({
                "iteration": 3 + i, "type": "tool_result",
                "tool": "search_knowledge_base",
                "result": sub_result[:300]
            })
            evidence_pieces.append((sub_q, sub_result))

        # Step 4: Synthesize final answer
        final_answer = synthesize_answer(question, evidence_pieces)

        num_iterations = 3 + len(sub_questions)
        reasoning_trace.append({
            "iteration": num_iterations,
            "type": "final_answer",
            "content": final_answer[:200]
        })

        return self._build_result(question, final_answer, reasoning_trace,
                                   num_iterations)

    # ------------------------------------------------------------------
    # Strategy: Semi-Structured (Wine Quality CSV)
    # ------------------------------------------------------------------
    def _run_semi_structured(self, question: str) -> dict:
        """Multi-step reasoning for structured data queries."""
        reasoning_trace = []
        evidence_pieces = []

        # Step 1: Get dataset overview
        summary = self.tool_executor.execute("get_document_summary", {})
        reasoning_trace.append({
            "iteration": 1, "type": "tool_call",
            "tool": "get_document_summary", "arguments": {}
        })
        reasoning_trace.append({
            "iteration": 1, "type": "tool_result",
            "tool": "get_document_summary", "result": summary[:300]
        })

        # Step 2: Run summary statistics
        stats = self.tool_executor.execute(
            "analyze_data", {"analysis_type": "summary_statistics"}
        )
        reasoning_trace.append({
            "iteration": 2, "type": "tool_call",
            "tool": "analyze_data",
            "arguments": {"analysis_type": "summary_statistics"}
        })
        reasoning_trace.append({
            "iteration": 2, "type": "tool_result",
            "tool": "analyze_data", "result": stats[:300]
        })
        evidence_pieces.append(("summary statistics", stats))

        # Step 3: Targeted analysis based on question keywords
        target_col = self._extract_column_hint(question)
        if target_col:
            # Run correlation analysis
            corr = self.tool_executor.execute(
                "analyze_data",
                {"analysis_type": "correlation", "target_column": target_col}
            )
            reasoning_trace.append({
                "iteration": 3, "type": "tool_call",
                "tool": "analyze_data",
                "arguments": {"analysis_type": "correlation",
                              "target_column": target_col}
            })
            reasoning_trace.append({
                "iteration": 3, "type": "tool_result",
                "tool": "analyze_data", "result": corr[:300]
            })
            evidence_pieces.append((f"correlation with {target_col}", corr))

            # Run distribution analysis
            dist = self.tool_executor.execute(
                "analyze_data",
                {"analysis_type": "distribution", "target_column": target_col}
            )
            reasoning_trace.append({
                "iteration": 4, "type": "tool_call",
                "tool": "analyze_data",
                "arguments": {"analysis_type": "distribution",
                              "target_column": target_col}
            })
            reasoning_trace.append({
                "iteration": 4, "type": "tool_result",
                "tool": "analyze_data", "result": dist[:300]
            })
            evidence_pieces.append((f"distribution of {target_col}", dist))

        # Step 4: Text search for additional context
        search_result = self.tool_executor.execute(
            "search_knowledge_base", {"query": question, "num_results": 3}
        )
        iteration_num = 5 if target_col else 3
        reasoning_trace.append({
            "iteration": iteration_num, "type": "tool_call",
            "tool": "search_knowledge_base",
            "arguments": {"query": question}
        })
        reasoning_trace.append({
            "iteration": iteration_num, "type": "tool_result",
            "tool": "search_knowledge_base",
            "result": search_result[:300]
        })
        evidence_pieces.append(("text search", search_result))

        # Synthesize
        final_answer = synthesize_answer(question, evidence_pieces)

        num_iterations = iteration_num
        reasoning_trace.append({
            "iteration": num_iterations + 1,
            "type": "final_answer",
            "content": final_answer[:200]
        })

        return self._build_result(question, final_answer, reasoning_trace,
                                   num_iterations)

    # ------------------------------------------------------------------
    # Strategy: Multi-Hop QA (HotpotQA)
    # ------------------------------------------------------------------
    def _run_multi_hop(self, question: str) -> dict:
        """Multi-step reasoning for multi-hop questions."""
        reasoning_trace = []
        evidence_pieces = []

        # Step 1: Direct search with original question
        search_result = self.tool_executor.execute(
            "search_knowledge_base", {"query": question, "num_results": 5}
        )
        reasoning_trace.append({
            "iteration": 1, "type": "tool_call",
            "tool": "search_knowledge_base",
            "arguments": {"query": question}
        })
        reasoning_trace.append({
            "iteration": 1, "type": "tool_result",
            "tool": "search_knowledge_base",
            "result": search_result[:300]
        })
        evidence_pieces.append((question, search_result))

        # Step 2: Decompose the question
        sub_questions = decompose_question(question)

        # Step 3: Search for each sub-question
        for i, sub_q in enumerate(sub_questions):
            sub_result = self.tool_executor.execute(
                "search_knowledge_base", {"query": sub_q, "num_results": 5}
            )
            reasoning_trace.append({
                "iteration": 2 + i, "type": "tool_call",
                "tool": "search_knowledge_base",
                "arguments": {"query": sub_q}
            })
            reasoning_trace.append({
                "iteration": 2 + i, "type": "tool_result",
                "tool": "search_knowledge_base",
                "result": sub_result[:300]
            })
            evidence_pieces.append((sub_q, sub_result))

        # Step 4: Explicit reasoning step
        reasoning_summary = f"Found {len(evidence_pieces)} pieces of evidence."
        reason_result = self.tool_executor.execute(
            "reason_step", {
                "current_findings": reasoning_summary,
                "reasoning": f"Combining evidence from {len(sub_questions)} sub-questions",
                "next_action": "Synthesize final answer"
            }
        )
        reason_iter = 2 + len(sub_questions)
        reasoning_trace.append({
            "iteration": reason_iter, "type": "tool_call",
            "tool": "reason_step",
            "arguments": {"current_findings": reasoning_summary}
        })
        reasoning_trace.append({
            "iteration": reason_iter, "type": "tool_result",
            "tool": "reason_step", "result": reason_result[:300]
        })

        # Step 5: Synthesize final answer
        final_answer = synthesize_answer(question, evidence_pieces)

        num_iterations = reason_iter + 1
        reasoning_trace.append({
            "iteration": num_iterations,
            "type": "final_answer",
            "content": final_answer[:200]
        })

        return self._build_result(question, final_answer, reasoning_trace,
                                   num_iterations)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_column_hint(self, question: str) -> str:
        """Extract a likely column name from the question for data analysis."""
        column_keywords = {
            "alcohol": "alcohol",
            "ph": "pH",
            "quality": "quality",
            "acidity": "volatile acidity",
            "volatile acidity": "volatile acidity",
            "fixed acidity": "fixed acidity",
            "citric acid": "citric acid",
            "sugar": "residual sugar",
            "residual sugar": "residual sugar",
            "chloride": "chlorides",
            "sulfur": "total sulfur dioxide",
            "sulfate": "sulphates",
            "density": "density",
        }
        q_lower = question.lower()
        for keyword, col in column_keywords.items():
            if keyword in q_lower:
                return col
        return "quality"  # Default

    def _build_result(self, question: str, answer: str,
                      reasoning_trace: list, num_iterations: int) -> dict:
        """Build a standardized result dict."""
        tool_breakdown = self.tool_executor.get_tool_call_breakdown()
        total_tool_calls = self.tool_executor.get_tool_call_count()

        # Calculate reasoning depth
        search_count = tool_breakdown.get("search_knowledge_base", 0)
        reason_count = tool_breakdown.get("reason_step", 0)
        analyze_count = tool_breakdown.get("analyze_data", 0)
        reasoning_depth = search_count + reason_count + analyze_count

        return {
            "question": question,
            "answer": answer,
            "reasoning_trace": reasoning_trace,
            "num_iterations": num_iterations,
            "total_tool_calls": total_tool_calls,
            "tool_call_breakdown": tool_breakdown,
            "reasoning_depth": reasoning_depth,
        }
