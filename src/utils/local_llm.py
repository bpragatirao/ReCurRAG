"""
Local LLM — Free, local text generation using HuggingFace Transformers.

Uses google/flan-t5-base for answer generation. Runs entirely on CPU,
requires no API keys or billing.

Model: google/flan-t5-base (~250MB, downloads automatically on first use)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Global model cache (loaded once, reused across calls)
_model = None
_tokenizer = None
_MODEL_NAME = "google/flan-t5-base"


def get_model():
    """Load and cache the local model. Downloads on first use."""
    global _model, _tokenizer

    if _model is None:
        print(f"  📥 Loading local model: {_MODEL_NAME} (first time may take a minute)...")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
        _model.eval()
        print(f"  ✅ Model loaded successfully!")

    return _model, _tokenizer


def generate_text(prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate text using the local model.

    Args:
        prompt: The input prompt.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Generated text string.
    """
    model, tokenizer = get_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()


def answer_question(question: str, context: str) -> str:
    """
    Answer a question given context, using the local model.

    Args:
        question: The question to answer.
        context: The relevant context to base the answer on.

    Returns:
        Generated answer string.
    """
    # Truncate context to fit within model limits
    max_context_chars = 1500
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."

    prompt = (
        f"Answer the question based only on the following context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return generate_text(prompt, max_new_tokens=256)


def decompose_question(question: str) -> list:
    """
    Break a multi-hop question into simpler sub-questions.

    Args:
        question: The complex question to decompose.

    Returns:
        List of simpler sub-questions.
    """
    prompt = (
        f"Break this question into 2-3 simpler sub-questions that can be "
        f"answered independently:\n\n"
        f"Question: {question}\n\n"
        f"Sub-questions:"
    )

    result = generate_text(prompt, max_new_tokens=200)

    # Parse the result into individual sub-questions
    sub_questions = []
    for line in result.split("\n"):
        line = line.strip()
        # Remove numbering prefixes like "1.", "1)", "-", etc.
        if line:
            for prefix in ["1.", "2.", "3.", "1)", "2)", "3)", "-", "•", "*"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line and len(line) > 5:
                sub_questions.append(line)

    # If decomposition failed, use the original question with slight variations
    if not sub_questions:
        sub_questions = [question]

    return sub_questions[:3]  # Max 3 sub-questions


def synthesize_answer(question: str, evidence_pieces: list) -> str:
    """
    Synthesize a final answer from multiple pieces of evidence.

    Args:
        question: The original question.
        evidence_pieces: List of (sub_question, evidence_text) tuples.

    Returns:
        Synthesized answer string.
    """
    evidence_text = ""
    for i, (sub_q, evidence) in enumerate(evidence_pieces):
        evidence_text += f"\nFinding {i+1} (about: {sub_q}):\n{evidence[:500]}\n"

    # Truncate if too long
    if len(evidence_text) > 1500:
        evidence_text = evidence_text[:1500] + "..."

    prompt = (
        f"Based on the following findings, provide a comprehensive answer "
        f"to the question.\n\n"
        f"Question: {question}\n\n"
        f"Findings:{evidence_text}\n\n"
        f"Answer:"
    )

    return generate_text(prompt, max_new_tokens=512)
