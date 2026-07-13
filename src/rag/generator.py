"""Question + retrieved chunks -> generated answer.

Two backends share the same generate_answer() interface, selected via
config.yaml's generation.backend:
  - "local": a small instruction-tuned seq2seq model (flan-t5-base) run
    on-device via `transformers`. No API key and no per-call network
    cost, so the pipeline works end-to-end with zero external accounts.
    This is the default.
  - "openai": stubbed for now. Swapping to a hosted model later (e.g. if
    verification/ ends up needing stronger generation) means overriding
    generation.backend in config.yaml and filling in OPENAI_API_KEY --
    not touching any other module, since callers only ever see
    generate_answer(question, context_chunks) -> str.
"""

from functools import lru_cache

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import get_config

PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say \"I don't know.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


@lru_cache(maxsize=1)
def _get_local_model():
    """Loads flan-t5 directly with AutoModelForSeq2SeqLM/AutoTokenizer
    rather than `transformers.pipeline("text2text-generation", ...)` --
    the pipeline task-string registry changes across transformers
    versions (it dropped that task name in newer releases), while the
    Auto* classes + .generate() are stable.
    """
    cfg = get_config()["generation"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["local_model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["local_model_name"])
    return tokenizer, model


def _generate_local(question: str, context: str, max_new_tokens: int) -> str:
    tokenizer, model = _get_local_model()
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _generate_openai(question: str, context: str, max_new_tokens: int) -> str:
    raise NotImplementedError(
        "OpenAI backend not implemented yet -- this is a placeholder for "
        "when a stronger generator is needed. To use it: implement this "
        "function with the `openai` package reading OPENAI_API_KEY from "
        "the environment, then set generation.backend: 'openai' in "
        "config.yaml. Until then, keep generation.backend: 'local'."
    )


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Generates an answer to `question` grounded in context_chunks.

    context_chunks is the list of dicts returned by retriever.retrieve()
    (each with a "text" key); only chunk["text"] is used here.
    """
    cfg = get_config()["generation"]
    context = "\n\n".join(chunk["text"] for chunk in context_chunks)

    if cfg["backend"] == "local":
        return _generate_local(question, context, cfg["max_new_tokens"])
    elif cfg["backend"] == "openai":
        return _generate_openai(question, context, cfg["max_new_tokens"])
    else:
        raise ValueError(f"Unknown generation backend: {cfg['backend']!r}")
