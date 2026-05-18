"""
RLM Tools — Dynamic tools available to the Recursive Language Model agent.

These tools allow the agent to iteratively search, analyze, and reason over
the data, unlike RAG which performs a single retrieval pass.

Tools:
    1. search_knowledge_base  — Semantic search over the FAISS vector store
    2. analyze_data           — Statistical analysis on structured data
    3. get_document_summary   — Get a summary/overview of loaded documents
    4. reason_step            — Explicit multi-hop reasoning step
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Tool Definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the knowledge base for information relevant to a query. "
                "Use this tool to find specific facts, data points, or passages "
                "from the ingested documents. You can call this multiple times "
                "with different queries to gather information from different angles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Perform statistical analysis or data inspection on the ingested data. "
                "Use this for structured/tabular data to compute averages, correlations, "
                "distributions, or other quantitative insights. Specify the analysis type "
                "and any relevant parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["summary_statistics", "correlation", "distribution",
                                 "comparison", "filter", "custom"],
                        "description": "The type of analysis to perform."
                    },
                    "target_column": {
                        "type": "string",
                        "description": "The column or feature to analyze (if applicable)."
                    },
                    "condition": {
                        "type": "string",
                        "description": "A filtering condition (e.g., 'quality > 7')."
                    }
                },
                "required": ["analysis_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_summary",
            "description": (
                "Get an overview of the loaded documents including source names, "
                "sizes, and key metadata. Use this to understand what data is "
                "available before performing specific searches."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reason_step",
            "description": (
                "Perform an explicit reasoning step. Use this to combine information "
                "from previous searches, draw intermediate conclusions, or plan the "
                "next steps in multi-hop reasoning. Provide your current findings "
                "and what you still need to determine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_findings": {
                        "type": "string",
                        "description": "Summary of what you have found so far."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning based on current findings."
                    },
                    "next_action": {
                        "type": "string",
                        "description": "What you plan to do next to complete the answer."
                    }
                },
                "required": ["current_findings", "reasoning"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool Execution Engine
# ---------------------------------------------------------------------------

class ToolExecutor:
    """
    Executes tools called by the RLM agent.

    Maintains references to the vector store and raw documents so tools
    can dynamically access them during the agent's reasoning loop.
    """

    def __init__(self, vector_store, documents: list, chunks: list,
                 data_type: str = "long_docs"):
        self.vector_store = vector_store
        self.documents = documents
        self.chunks = chunks
        self.data_type = data_type
        self.tool_call_log = []  # Track all tool calls for analysis

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dict of tool arguments.

        Returns:
            String result from the tool execution.
        """
        # Log the tool call
        self.tool_call_log.append({
            "tool": tool_name,
            "arguments": arguments
        })

        if tool_name == "search_knowledge_base":
            return self._search_knowledge_base(**arguments)
        elif tool_name == "analyze_data":
            return self._analyze_data(**arguments)
        elif tool_name == "get_document_summary":
            return self._get_document_summary()
        elif tool_name == "reason_step":
            return self._reason_step(**arguments)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _search_knowledge_base(self, query: str, num_results: int = 5) -> str:
        """Semantic search over the FAISS vector store."""
        try:
            docs = self.vector_store.similarity_search(query, k=num_results)
            results = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "unknown")
                results.append(
                    f"[Result {i+1}] (Source: {source})\n{doc.page_content}"
                )
            return "\n\n".join(results) if results else "No relevant results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    def _analyze_data(self, analysis_type: str, target_column: str = None,
                      condition: str = None) -> str:
        """Perform data analysis on structured datasets."""
        if self.data_type != "semi_structured":
            # For non-tabular data, fall back to a search-based analysis
            return self._search_knowledge_base(
                f"statistical analysis {target_column or ''} {condition or ''}"
            )

        try:
            # Load the actual CSV data for real analysis
            data_dir = None
            for doc in self.documents:
                if doc["source"].endswith(".csv"):
                    # Reconstruct path from source
                    data_dir = "data/raw/Semi-Structured/wine+quality/"
                    break

            if not data_dir:
                return "No structured data available for analysis."

            results = []
            csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

            for csv_file in csv_files:
                df = pd.read_csv(os.path.join(data_dir, csv_file), sep=";")
                results.append(f"\n--- {csv_file} ---")

                if analysis_type == "summary_statistics":
                    if target_column and target_column in df.columns:
                        results.append(df[target_column].describe().to_string())
                    else:
                        results.append(df.describe().to_string())

                elif analysis_type == "correlation":
                    if target_column and target_column in df.columns:
                        corr = df.corr()[target_column].sort_values(ascending=False)
                        results.append(f"Correlations with '{target_column}':\n{corr.to_string()}")
                    else:
                        results.append(df.corr().to_string())

                elif analysis_type == "distribution":
                    if target_column and target_column in df.columns:
                        dist = df[target_column].value_counts().sort_index()
                        results.append(f"Distribution of '{target_column}':\n{dist.to_string()}")
                        results.append(f"\nMean: {df[target_column].mean():.4f}")
                        results.append(f"Median: {df[target_column].median():.4f}")
                        results.append(f"Std: {df[target_column].std():.4f}")
                    else:
                        results.append(f"Available columns: {', '.join(df.columns.tolist())}")

                elif analysis_type == "comparison":
                    if target_column and target_column in df.columns:
                        grouped = df.groupby("quality")[target_column].agg(
                            ["mean", "std", "min", "max", "count"]
                        )
                        results.append(
                            f"'{target_column}' grouped by quality:\n{grouped.to_string()}"
                        )
                    else:
                        results.append(
                            f"Available columns: {', '.join(df.columns.tolist())}"
                        )

                elif analysis_type == "filter":
                    if condition:
                        try:
                            filtered = df.query(condition)
                            results.append(
                                f"Filtered ({condition}): {len(filtered)} records out of {len(df)}"
                            )
                            results.append(filtered.describe().to_string())
                        except Exception as e:
                            results.append(f"Filter error: {str(e)}")
                    else:
                        results.append("No filter condition provided.")

                elif analysis_type == "custom":
                    results.append(f"Dataset shape: {df.shape}")
                    results.append(f"Columns: {', '.join(df.columns.tolist())}")
                    results.append(f"\nFirst 5 rows:\n{df.head().to_string()}")

            return "\n".join(results)

        except Exception as e:
            return f"Analysis error: {str(e)}"

    def _get_document_summary(self) -> str:
        """Provide an overview of loaded documents."""
        summary_parts = [
            f"Data Type: {self.data_type}",
            f"Total Documents: {len(self.documents)}",
            f"Total Chunks: {len(self.chunks)}",
            "\nDocument Details:"
        ]

        for doc in self.documents[:20]:  # Limit to first 20
            source = doc.get("source", "unknown")
            content_len = len(doc.get("content", ""))
            extra = ""
            if "question" in doc:
                extra = f" | Q: {doc['question'][:60]}..."
            summary_parts.append(f"  - {source}: {content_len:,} chars{extra}")

        if len(self.documents) > 20:
            summary_parts.append(f"  ... and {len(self.documents) - 20} more documents")

        return "\n".join(summary_parts)

    def _reason_step(self, current_findings: str, reasoning: str,
                     next_action: str = None) -> str:
        """
        Process an explicit reasoning step.
        Returns a structured summary of the reasoning.
        """
        result = f"📝 Reasoning Step:\n"
        result += f"  Findings: {current_findings}\n"
        result += f"  Reasoning: {reasoning}\n"
        if next_action:
            result += f"  Next Action: {next_action}\n"
        return result

    def get_tool_call_count(self) -> int:
        """Return the total number of tool calls made."""
        return len(self.tool_call_log)

    def get_tool_call_breakdown(self) -> dict:
        """Return a breakdown of tool calls by type."""
        breakdown = {}
        for call in self.tool_call_log:
            tool = call["tool"]
            breakdown[tool] = breakdown.get(tool, 0) + 1
        return breakdown
