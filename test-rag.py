from src.rag.pipeline import RAGPipeline
import pandas as pd
import json

ragld = RAGPipeline(data_path="data/raw/Long-Docs/papers/")
ragld.ingest()

# ragss = RAGPipeline(data_path="data/raw/Semi-Strcutured/wine+quality/")
# ragss.ingest()

# with open("data/raw/Multi-HopQA/hotpotqa.json") as f:
#     data = json.load(f)

# df = pd.DataFrame(data)
# df.to_csv("data/processed/Multi-HopQA/hotpotqa.csv", index=False)
# ragmh = RAGPipeline(data_path="data/processed/Multi-HopQA/")
# ragmh.ingest()

query = "What is working capital and what does negative working capital mean?"

# response = {ragld.query(query), ragmh.query(query), ragss.query(query)}
response = ragld.query(query)

print("\n--- QUERY ---")
print(query)

print("\n--- RESPONSE ---")
print(response)

print("\nAnswer:\n", response)