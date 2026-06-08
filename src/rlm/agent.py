"""
RLM Agent — Recursive Language Model agent with iterative multi-step reasoning.

Uses a SCRIPTED tool-calling strategy with a local model (google/flan-t5-base).
No API keys or billing required.

Unlike RAG (single retrieve → generate), the RLM agent:
  1. Extracts key entities from the question
  2. Searches with the original query and entity-specific queries
  3. Extracts intermediate answers from each search
  4. Decomposes complex questions into sub-questions
  5. Performs data analysis when appropriate (analyze_data)
  6. Synthesizes all evidence with verification
"""

from .tools import ToolExecutor
from ..utils.local_llm import (
    answer_question,
    extract_answer_from_context,
    extract_entities,
    decompose_question,
    synthesize_answer,
    verify_answer,
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
          - multi_hop:       entities → search each → decompose → intermediate answers → verify

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

        # Step 3: Extract entities and search each
        entities = extract_entities(question)
        for i, entity in enumerate(entities[:2]):
            entity_result = self.tool_executor.execute(
                "search_knowledge_base",
                {"query": f"{entity} in the context of {question[:50]}", "num_results": 3}
            )
            reasoning_trace.append({
                "iteration": 3 + i, "type": "tool_call",
                "tool": "search_knowledge_base",
                "arguments": {"query": entity}
            })
            reasoning_trace.append({
                "iteration": 3 + i, "type": "tool_result",
                "tool": "search_knowledge_base",
                "result": entity_result[:300]
            })
            evidence_pieces.append((entity, entity_result))

        # Step 4: Decompose and search sub-questions
        sub_questions = decompose_question(question)
        for i, sub_q in enumerate(sub_questions[:2]):
            sub_result = self.tool_executor.execute(
                "search_knowledge_base", {"query": sub_q, "num_results": 3}
            )
            iter_num = 3 + len(entities[:2]) + i
            reasoning_trace.append({
                "iteration": iter_num, "type": "tool_call",
                "tool": "search_knowledge_base",
                "arguments": {"query": sub_q}
            })
            reasoning_trace.append({
                "iteration": iter_num, "type": "tool_result",
                "tool": "search_knowledge_base",
                "result": sub_result[:300]
            })
            evidence_pieces.append((sub_q, sub_result))

        # Step 5: Synthesize final answer
        final_answer = synthesize_answer(question, evidence_pieces)

        num_iterations = 3 + len(entities[:2]) + len(sub_questions[:2])
        reasoning_trace.append({
            "iteration": num_iterations + 1,
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
    # Strategy: Multi-Hop QA (HotpotQA) — IMPROVED
    # ------------------------------------------------------------------
    def _run_multi_hop(self, question: str) -> dict:
        """
        Enhanced multi-step reasoning for multi-hop questions.

        Strategy:
          1. Extract entities from the question
          2. Search for the full question
          3. Search for each entity separately
          4. Extract intermediate answers from each search
          5. Decompose into sub-questions and search those too
          6. Synthesize from all evidence
          7. Verify the answer against the best context
        """
        reasoning_trace = []
        evidence_pieces = []
        all_context = []

        # Step 1: Extract key entities
        entities = extract_entities(question)
        reasoning_trace.append({
            "iteration": 1, "type": "entity_extraction",
            "entities": entities
        })

        # Step 2: Direct search with original question
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
        all_context.append(search_result)

        # Step 3: Entity-specific searches
        for i, entity in enumerate(entities[:3]):
            entity_result = self.tool_executor.execute(
                "search_knowledge_base", {"query": entity, "num_results": 5}
            )
            iter_num = 3 + i
            reasoning_trace.append({
                "iteration": iter_num, "type": "tool_call",
                "tool": "search_knowledge_base",
                "arguments": {"query": entity}
            })
            reasoning_trace.append({
                "iteration": iter_num, "type": "tool_result",
                "tool": "search_knowledge_base",
                "result": entity_result[:300]
            })
            evidence_pieces.append((f"About {entity}", entity_result))
            all_context.append(entity_result)

        # Step 4: Extract intermediate answer from combined context
        combined_context = "\n\n".join(all_context[:3])  # Use top 3 contexts
        intermediate = extract_answer_from_context(question, combined_context)
        reasoning_trace.append({
            "iteration": 3 + len(entities[:3]),
            "type": "intermediate_answer",
            "content": intermediate
        })

        # Step 5: Decompose and search sub-questions
        sub_questions = decompose_question(question)
        sub_iter_start = 4 + len(entities[:3])
        for i, sub_q in enumerate(sub_questions[:2]):
            sub_result = self.tool_executor.execute(
                "search_knowledge_base", {"query": sub_q, "num_results": 3}
            )
            iter_num = sub_iter_start + i
            reasoning_trace.append({
                "iteration": iter_num, "type": "tool_call",
                "tool": "search_knowledge_base",
                "arguments": {"query": sub_q}
            })
            reasoning_trace.append({
                "iteration": iter_num, "type": "tool_result",
                "tool": "search_knowledge_base",
                "result": sub_result[:300]
            })
            evidence_pieces.append((sub_q, sub_result))
            all_context.append(sub_result)

        # Step 6: Reasoning step — combine evidence
        reasoning_summary = (
            f"Found {len(evidence_pieces)} pieces of evidence from "
            f"{len(entities)} entities. Intermediate answer: {intermediate}"
        )
        reason_result = self.tool_executor.execute(
            "reason_step", {
                "current_findings": reasoning_summary,
                "reasoning": f"Combining evidence across {len(entities)} entities and {len(sub_questions)} sub-questions",
                "next_action": "Synthesize and verify final answer"
            }
        )
        reason_iter = sub_iter_start + len(sub_questions[:2])
        reasoning_trace.append({
            "iteration": reason_iter, "type": "tool_call",
            "tool": "reason_step",
            "arguments": {"current_findings": reasoning_summary[:200]}
        })
        reasoning_trace.append({
            "iteration": reason_iter, "type": "tool_result",
            "tool": "reason_step", "result": reason_result[:300]
        })

        # Step 7: Synthesize final answer
        final_answer = synthesize_answer(question, evidence_pieces)

        # Step 8: Verify with the best context
        best_context = "\n".join(all_context[:2])
        verified = verify_answer(question, final_answer, best_context)
        verify_iter = reason_iter + 1
        reasoning_trace.append({
            "iteration": verify_iter, "type": "verification",
            "proposed": final_answer[:100],
            "verified": verified[:100]
        })

        # Use verified answer if it's substantive
        if verified and len(verified.strip()) > 1 and "yes" not in verified.lower()[:5]:
            final_answer = verified

        num_iterations = verify_iter
        reasoning_trace.append({
            "iteration": num_iterations + 1,
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
        summary_count = tool_breakdown.get("get_document_summary", 0)
        reasoning_depth = search_count + reason_count + analyze_count + summary_count

        return {
            "question": question,
            "answer": answer,
            "reasoning_trace": reasoning_trace,
            "num_iterations": num_iterations,
            "total_tool_calls": total_tool_calls,
            "tool_call_breakdown": tool_breakdown,
            "reasoning_depth": reasoning_depth,
        }
