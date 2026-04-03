# 🚀 ReCurRAG: From Retrieval to Recursive Reasoning

ReCurRAG is a research-driven project comparing Retrieval-Augmented Generation (RAG) with Recursive Language Models (RLMs) on long-context and multi-hop reasoning tasks. This project demonstrates why retrieval-based systems struggle with complex reasoning and how recursive agent-based models provide deeper, more reliable intelligence.

This project serves as a comprehensive benchmarking framework to evaluate the performance gap between traditional **Retrieval-Augmented Generation (RAG)** and **Recursive Language Models (Agent-based reasoning systems)**.

## 📌 Project Overview
ReCurRAG is designed to test how different architectures handle complex data retrieval and synthesis tasks. While standard RAG is efficient for simple queries, it often struggles with "global" understanding and multi-step logic. This project implements a Recursive LM approach to bridge those gaps.

### 🧠 Motivation
* **Traditional RAG Systems**: These typically retrieve the "top-k" most relevant chunks based on semantic similarity. However, they often lose global context and fail at multi-step reasoning.
* **Recursive Language Models**: These systems explore data iteratively, using tools dynamically to perform multi-hop reasoning, allowing for a deeper understanding of the dataset.

## 🏗️ Architecture Comparison

### 🔹 Standard RAG
**Workflow**: `Query → Retrieve → Generate`
A linear process that relies on the initial retrieval step to provide all necessary information for the final answer.

### 🔹 Recursive LM (Agentic)
**Workflow**: `Query → Plan → Tool Use → Reason → Refine → Aggregate`
A dynamic, looping process where the model can "think," use tools to find more data, and refine its answer based on new findings.

## 📂 Dataset Architecture
The project evaluates these systems across a diverse three-tier dataset structure:
1.  **Long Documents (Unstructured)**: The **Indian Constitution (PDF)** and **arXiv Research Papers** are used to test long-context retrieval and summarization capabilities.
2.  **Structured Data**: **World Bank CSVs** and **UCI Machine Learning Repository** files (like Wine Quality) are used to test tabular reasoning.
3.  **Multi-hop QA**: **HotpotQA** serves as the primary benchmark for testing explainable, multi-step logical chains.

## 📊 Evaluation Metrics
To measure success, the project tracks several key performance indicators:
* **Exact Match (EM)** & **F1 Score**: Standard accuracy metrics.
* **Reasoning Depth**: How many "hops" or logical steps the model successfully navigated.
* **Context Coverage**: The percentage of relevant information from the source documents included in the final output.

## 📈 Benchmarking Results (Expected)

| Capability | Standard RAG | Recursive LM |
| :--- | :---: | :---: |
| **Long Context** | ❌ | ✅ |
| **Multi-hop Reasoning** | ❌ | ✅ |
| **Context Completeness** | ❌ | ✅ |

## ⚙️ Setup & Execution

### Installation
```bash
git clone [https://github.com/bpragatirao/ReCurRAG.git](https://github.com/bpragatirao/ReCurRAG.git)
cd ReCurRAG
pip install -r requirements.txt
```
