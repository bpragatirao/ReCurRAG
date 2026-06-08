"""
Local LLM — Free, local text generation using HuggingFace Transformers.

Uses google/flan-t5-base for answer generation. Runs entirely on CPU,
requires no API keys or billing.

Model: google/flan-t5-base (~250MB, downloads automatically on first use)
"""

import re
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


def extract_answer_from_context(question: str, context: str) -> str:
    """
    Extract a precise, short answer from context.
    Uses a more focused extractive prompt for better accuracy.

    Args:
        question: The question to answer.
        context: The relevant context.

    Returns:
        Extracted answer string.
    """
    max_context_chars = 1200
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."

    prompt = (
        f"Extract the precise answer to the question from the context. "
        f"Give only the answer, nothing else.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return generate_text(prompt, max_new_tokens=64)


def extract_entities(question: str) -> list:
    """
    Extract key named entities and noun phrases from a question.
    Uses a combination of pattern matching and model-based extraction.

    Args:
        question: The question to extract entities from.

    Returns:
        List of entity strings.
    """
    # First try model-based extraction
    prompt = (
        f"List the key entities (people, places, things, events) mentioned "
        f"in this question, separated by commas:\n\n"
        f"Question: {question}\n\n"
        f"Entities:"
    )

    result = generate_text(prompt, max_new_tokens=100)
    entities = [e.strip() for e in result.split(",") if e.strip() and len(e.strip()) > 2]

    # Fallback: extract quoted terms and capitalized words
    if not entities:
        # Get quoted strings
        quoted = re.findall(r'"([^"]+)"', question)
        entities.extend(quoted)

        # Get capitalized multi-word names (at least 2 chars, not starting words)
        words = question.split()
        for i, word in enumerate(words):
            clean = word.strip('?.,!;:()[]"\'')
            if clean and clean[0].isupper() and i > 0 and len(clean) > 2:
                entities.append(clean)

    return entities[:5]  # Max 5 entities


def decompose_question(question: str) -> list:
    """
    Break a multi-hop question into simpler sub-questions.
    Uses entity extraction to create targeted sub-questions.

    Args:
        question: The complex question to decompose.

    Returns:
        List of simpler sub-questions.
    """
    # Extract entities for targeted sub-questions
    entities = extract_entities(question)

    sub_questions = []

    # Create entity-based sub-questions
    for entity in entities[:3]:
        sub_q = f"What is {entity}?"
        sub_questions.append(sub_q)

    # Also try model-based decomposition
    prompt = (
        f"Break this complex question into 2 simpler questions:\n\n"
        f"Question: {question}\n\n"
        f"Simple questions:"
    )

    result = generate_text(prompt, max_new_tokens=150)

    for line in result.split("\n"):
        line = line.strip()
        if line:
            for prefix in ["1.", "2.", "3.", "1)", "2)", "3)", "-", "•", "*"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line and len(line) > 5 and line not in sub_questions:
                sub_questions.append(line)

    # Ensure we always have at least the original question
    if not sub_questions:
        sub_questions = [question]

    return sub_questions[:4]  # Max 4 sub-questions


def synthesize_answer(question: str, evidence_pieces: list) -> str:
    """
    Synthesize a final answer from multiple pieces of evidence.
    Uses a two-step approach: extract key facts, then synthesize.

    Args:
        question: The original question.
        evidence_pieces: List of (sub_question, evidence_text) tuples.

    Returns:
        Synthesized answer string.
    """
    # Step 1: Extract key facts from each evidence piece
    key_facts = []
    for sub_q, evidence in evidence_pieces:
        if not evidence or len(evidence.strip()) < 10:
            continue
        fact = extract_answer_from_context(sub_q, evidence)
        if fact and len(fact.strip()) > 1:
            key_facts.append(f"- {sub_q}: {fact}")

    facts_text = "\n".join(key_facts) if key_facts else "No clear facts extracted."

    # Step 2: Synthesize from extracted facts
    prompt = (
        f"Using these facts, answer the question concisely.\n\n"
        f"Facts:\n{facts_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return generate_text(prompt, max_new_tokens=256)


def verify_answer(question: str, answer: str, context: str) -> str:
    """
    Verify and potentially correct an answer using the context.

    Args:
        question: The original question.
        answer: The proposed answer.
        context: Supporting context.

    Returns:
        Verified/corrected answer.
    """
    max_context_chars = 800
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."

    prompt = (
        f"Given the context, is this answer correct? If not, provide the "
        f"correct answer.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        f"Correct answer:"
    )

    return generate_text(prompt, max_new_tokens=64)
