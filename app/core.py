"""
Support Response Generator - Core Engine
Implements text anonymization (PII redaction), smoothed n-gram language models,
autoregressive response generation, and agent assistant tools.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.storage import Store


# ==============================================================================
# Tokenization & Normalization Utilities
# ==============================================================================

def to_sentences(text: str) -> list[str]:
    """Split text into individual sentences using standard punctuation."""
    return [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]


def to_tokens(sentence: str) -> list[str]:
    """
    Extract tokens: lowercase words, numbers, contractions, and special tokens (e.g. [order_id]).
    """
    return re.findall(r"\[[a-z_]+\]|[a-z0-9']+", sentence.lower())


# ==============================================================================
# Milestone 1: Corpus Anonymization & Data Cleaning
# ==============================================================================

PII_PATTERNS = [
    # Emails
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[email]"),
    # Order / Return / Invoice / Item IDs
    (r"#?(ORD|ITM|RET|INV)-[0-9]{4,6}\b", "[order_id]"),
    # Carrier tracking numbers
    (r"\bTRK-[0-9]{8,12}\b", "[tracking_no]"),
    # Phone numbers
    (r"(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\+1[-.\s]?\d{3}[-.\s]?\d{4}\b", "[phone]"),
    # Currency amounts
    (r"\$\d+(?:\.\d{2})?\b", "[amount]"),
    # Dates
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "[date]"),
    # Customer name placeholders in greetings
    (r"\b(Hello|Dear|Hi)\s+[A-Z][a-z]+,", r"\1 customer,"),
]


def anonymize_text(text: str) -> tuple[str, dict[str, int]]:
    """
    Scrub PII from customer support interactions and replace with anonymized tokens.
    Returns (anonymized_text, redaction_counts).
    """
    counts: dict[str, int] = defaultdict(int)
    anonymized = text

    for pattern, replacement in PII_PATTERNS:
        matches = len(re.findall(pattern, anonymized, flags=re.IGNORECASE))
        if matches > 0:
            token_key = replacement if replacement.startswith("[") else "customer_name"
            counts[token_key] += matches
            anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)

    anonymized = re.sub(r"[ \t]+", " ", anonymized)
    return anonymized.strip(), dict(counts)


def prepare_corpus(raw_text: str) -> tuple[str, dict[str, Any]]:
    """
    Process raw support conversations into a clean, anonymized corpus.
    """
    anonymized, redactions = anonymize_text(raw_text)
    sents = to_sentences(anonymized)
    tokens = [t for s in sents for t in to_tokens(s)]

    stats = {
        "sentences": len(sents),
        "tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "redactions": redactions,
    }
    return anonymized, stats


def milestone_1(input_path: str | None = None, output_path: str | None = None) -> dict[str, Any]:
    """
    Milestone 1 implementation:
    Reads raw support interactions, scrubs all PII, saves anonymized corpus,
    and returns processing metrics.
    """
    in_file = Path(input_path or "data/raw_support_logs.txt")
    out_file = Path(output_path or "data/anonymized_corpus.txt")

    if not in_file.exists():
        raise FileNotFoundError(f"Source file not found: {in_file}")

    raw_content = in_file.read_text(encoding="utf-8")
    anonymized_content, stats = prepare_corpus(raw_content)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(anonymized_content, encoding="utf-8")

    return {
        "status": "success",
        "output_file": str(out_file),
        "stats": stats,
    }


# ==============================================================================
# Milestone 2: Smoothed N-gram Language Model
# ==============================================================================

def build_ngram_counts(sents: list[str], n: int = 2) -> dict[tuple[str, ...], Counter]:
    """
    Map context (length n-1) -> Counter of subsequent tokens.
    """
    foll = defaultdict(Counter)
    for s in sents:
        toks = ["<s>"] * (n - 1) + to_tokens(s) + ["</s>"]
        for i in range(len(toks) - n + 1):
            ctx = tuple(toks[i : i + n - 1])
            nxt = toks[i + n - 1]
            foll[ctx][nxt] += 1
    return foll


class SupportNGramLM:
    """
    Statistical N-gram Language Model designed for support response drafting.
    Includes Add-k (Laplace) smoothing, Jelinek-Mercer linear interpolation,
    and Hierarchical Backoff across context lengths (up to 4-grams).
    """

    def __init__(self, n: int = 4, k: float = 0.5, interpolation: bool = False):
        self.n = max(2, n)
        self.k = k
        self.use_interpolation = interpolation
        self.sents: list[str] = []
        self.order_counts: dict[int, dict[tuple[str, ...], Counter]] = {}
        self.foll: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.uni_counts: Counter = Counter()
        self.V: set[str] = set()

    def fit(self, text: str) -> SupportNGramLM:
        """Train multi-order n-gram model on the given text corpus."""
        self.sents = to_sentences(text)

        # Unigram counts
        self.uni_counts = Counter(t for s in self.sents for t in to_tokens(s))
        self.uni_counts["</s>"] = len(self.sents)
        self.V = set(self.uni_counts.keys()) | {"</s>", "<unk>"}

        # Build counts for all orders up to self.n
        for order in range(2, max(5, self.n + 1)):
            self.order_counts[order] = build_ngram_counts(self.sents, order)

        self.foll = self.order_counts.get(self.n, build_ngram_counts(self.sents, self.n))
        return self

    def get_backoff_counter(self, tokens: list[str]) -> tuple[tuple[str, ...], Counter]:
        """
        Hierarchical backoff: searches from largest context window down to unigram.
        """
        max_order = max(self.order_counts.keys()) if self.order_counts else self.n
        for order in range(max_order, 1, -1):
            ctx_len = order - 1
            ctx = tuple((["<s>"] * ctx_len + tokens)[-ctx_len:])
            counts_map = self.order_counts.get(order, {})
            if ctx in counts_map and counts_map[ctx]:
                return ctx, counts_map[ctx]

        return ("<s>",), Counter(self.uni_counts)

    def prob_addk(self, ctx: tuple[str, ...], w: str) -> float:
        """
        Add-k (Laplace) smoothed probability:
        P(w | ctx) = (count(ctx, w) + k) / (count(ctx) + k * |V|)
        """
        order = len(ctx) + 1
        counts_map = self.order_counts.get(order, self.foll)
        c = counts_map.get(ctx, Counter())
        total = sum(c.values())
        return (c[w] + self.k) / (total + self.k * len(self.V))

    def prob_interpolated(self, ctx: tuple[str, ...], w: str, lambdas: tuple[float, float, float] = (0.1, 0.3, 0.6)) -> float:
        """
        Jelinek-Mercer linear interpolation:
        P(w | w-2, w-1) = l1 * P_uni(w) + l2 * P_bi(w | w-1) + l3 * P_tri(w | w-2, w-1)
        """
        l1, l2, l3 = lambdas
        total_tokens = sum(self.uni_counts.values()) or 1
        p_uni = (self.uni_counts[w] + 1e-5) / (total_tokens + 1e-5 * len(self.V))

        # Bigram
        bi_map = self.order_counts.get(2, {})
        bi_ctx = (ctx[-1],) if ctx else ("<s>",)
        bi_c = bi_map.get(bi_ctx, Counter())
        bi_total = sum(bi_c.values())
        p_bi = (bi_c[w] + self.k) / (bi_total + self.k * len(self.V))

        # Trigram
        tri_map = self.order_counts.get(3, {})
        tri_ctx = ctx[-2:] if len(ctx) >= 2 else (("<s>",) * (2 - len(ctx)) + ctx)
        tri_c = tri_map.get(tri_ctx, Counter())
        tri_total = sum(tri_c.values())
        p_tri = (tri_c[w] + self.k) / (tri_total + self.k * len(self.V))

        return l1 * p_uni + l2 * p_bi + l3 * p_tri

    def prob(self, ctx: tuple[str, ...], w: str) -> float:
        """Get transition probability using model configuration."""
        if self.use_interpolation:
            return self.prob_interpolated(ctx, w)
        return self.prob_addk(ctx, w)

    def perplexity(self, text: str) -> float:
        """Compute cross-entropy perplexity on held-out text."""
        sents = to_sentences(text)
        log_sum, n_tokens = 0.0, 0

        for s in sents:
            toks = ["<s>"] * (self.n - 1) + to_tokens(s) + ["</s>"]
            for i in range(len(toks) - self.n + 1):
                ctx = tuple(toks[i : i + self.n - 1])
                w = toks[i + self.n - 1]
                p = self.prob(ctx, w)
                log_sum += math.log(p)
                n_tokens += 1

        return math.exp(-log_sum / n_tokens) if n_tokens > 0 else float("inf")


def milestone_2(corpus_path: str | None = None) -> dict[str, Any]:
    """
    Milestone 2 implementation:
    Initializes and trains Bigram and Trigram models on anonymized support corpus,
    verifying Laplace smoothing and perplexity calculation.
    """
    c_path = Path(corpus_path or "data/anonymized_corpus.txt")
    if not c_path.exists():
        milestone_1()

    corpus_text = c_path.read_text(encoding="utf-8")
    sents = to_sentences(corpus_text)

    rng = random.Random(42)
    shuffled = list(sents)
    rng.shuffle(shuffled)
    cut = int(0.8 * len(shuffled))
    train_text = "\n".join(shuffled[:cut])
    test_text = "\n".join(shuffled[cut:])

    bigram = SupportNGramLM(n=2, k=0.5).fit(train_text)
    trigram = SupportNGramLM(n=3, k=0.5).fit(train_text)

    ppl_bi_train = bigram.perplexity(train_text)
    ppl_bi_test = bigram.perplexity(test_text)
    ppl_tri_train = trigram.perplexity(train_text)
    ppl_tri_test = trigram.perplexity(test_text)

    return {
        "status": "success",
        "vocabulary_size": len(bigram.V),
        "bigram": {"train_ppl": round(ppl_bi_train, 2), "test_ppl": round(ppl_bi_test, 2)},
        "trigram": {"train_ppl": round(ppl_tri_train, 2), "test_ppl": round(ppl_tri_test, 2)},
    }


# ==============================================================================
# Milestone 3: Autocomplete Suggestions & Autoregressive Continuation
# ==============================================================================

def suggest_next_words(model: SupportNGramLM, text: str, top_k: int = 3) -> list[tuple[str, float]]:
    """
    Given partial text context, compute candidate next-word ranking with relative confidence.
    Uses hierarchical backoff for natural, context-aware predictions.
    """
    tokens = to_tokens(text)
    ctx, counter = model.get_backoff_counter(tokens)

    # Filter out sentence terminator for inline autocomplete suggestions
    valid_candidates = {w: count for w, count in counter.items() if w != "</s>"}
    if not valid_candidates:
        valid_candidates = {w: count for w, count in model.uni_counts.items() if w != "</s>"}

    total_valid = sum(valid_candidates.values()) or 1
    ranked = sorted(valid_candidates.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    return [(w, round(count / total_valid, 4)) for w, count in ranked]


def generate_continuation(
    model: SupportNGramLM,
    prompt: str,
    max_tokens: int = 25,
    seed: int | None = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """
    Autoregressively generate next words following the prompt using hierarchical backoff.
    Default temperature 0.0 uses greedy argmax for coherent, professional customer support sentences.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    tokens = to_tokens(prompt)
    generated_tokens: list[str] = []

    for _ in range(max_tokens):
        current_tokens = tokens + generated_tokens
        ctx, c = model.get_backoff_counter(current_tokens)
        if not c:
            break

        words, counts = list(zip(*c.items()))
        words, counts = list(words), list(counts)

        # Greedy argmax (standard for accurate support response drafting)
        if temperature <= 0.05:
            best_idx = max(range(len(counts)), key=lambda i: counts[i])
            nxt = words[best_idx]
        else:
            # Temperature scaling
            adj_weights = [math.pow(max(cnt, 1e-4), 1.0 / max(temperature, 0.1)) for cnt in counts]
            total_wt = sum(adj_weights)
            probs = [w / total_wt for w in adj_weights]

            # Top-p (nucleus) filtering
            sorted_pairs = sorted(zip(words, probs), key=lambda x: x[1], reverse=True)
            filtered_words, filtered_weights = [], []
            cum = 0.0
            for w, pr in sorted_pairs:
                filtered_words.append(w)
                filtered_weights.append(pr)
                cum += pr
                if cum >= top_p:
                    break
            nxt = rng.choices(filtered_words, weights=filtered_weights)[0]

        if nxt == "</s>":
            break

        generated_tokens.append(nxt)

    completed = (prompt.strip() + " " + " ".join(generated_tokens)).strip()
    if not completed.endswith((".", "!", "?")):
        completed += "."
    return completed


def milestone_3(model: SupportNGramLM | None = None) -> dict[str, Any]:
    """
    Milestone 3 implementation:
    Demonstrates autocomplete suggestions and response continuation on support prompts.
    """
    if model is None:
        c_path = Path("data/anonymized_corpus.txt")
        if not c_path.exists():
            milestone_1()
        model = SupportNGramLM(n=4, k=0.5).fit(c_path.read_text(encoding="utf-8"))

    test_prompts = [
        "thank you for contacting",
        "your order has been",
        "we have processed your",
        "we apologize for the",
    ]

    suggestions = {p: suggest_next_words(model, p, top_k=3) for p in test_prompts}
    continuations = {p: generate_continuation(model, p, max_tokens=15, seed=42) for p in test_prompts}

    return {
        "status": "success",
        "suggestions": suggestions,
        "continuations": continuations,
    }


# ==============================================================================
# Milestone 4: Command-Line Interface Logic & Agent Interaction
# ==============================================================================

def format_suggestions_table(prompt: str, suggestions: list[tuple[str, float]]) -> str:
    """Format autocomplete suggestions for terminal display."""
    lines = [f"Prompt: \"{prompt}\"", "Next-word suggestions:"]
    for idx, (word, prob) in enumerate(suggestions, 1):
        confidence = f"{prob * 100:5.1f}%"
        bar = "█" * max(1, int(prob * 20))
        lines.append(f"  [{idx}] {word:<15} (confidence: {confidence}) {bar}")
    return "\n".join(lines)


def milestone_4(prompt: str = "thank you for contacting customer") -> dict[str, Any]:
    """
    Milestone 4 implementation:
    Validates end-to-end CLI suggestion pipeline.
    """
    c_path = Path("data/anonymized_corpus.txt")
    if not c_path.exists():
        milestone_1()
    model = SupportNGramLM(n=4, k=0.5).fit(c_path.read_text(encoding="utf-8"))

    suggs = suggest_next_words(model, prompt, top_k=3)
    completed = generate_continuation(model, prompt, max_tokens=15, seed=0)
    formatted = format_suggestions_table(prompt, suggs)

    return {
        "status": "success",
        "prompt": prompt,
        "suggestions": suggs,
        "completion": completed,
        "formatted_display": formatted,
    }


# ==============================================================================
# Stretch Goals: Multi-Corpus Switcher, Feedback, and Canned Templates
# ==============================================================================

class SupportAssistant:
    """
    High-level Support Assistant managing domain corpora, response templates,
    and agent feedback tracking with Hierarchical Backoff.
    """

    def __init__(self, data_path: str = "data/app.json"):
        self.store = Store(data_path)
        self.models: dict[str, SupportNGramLM] = {}
        self._init_corpora()
        self._init_default_templates()

    def _init_corpora(self) -> None:
        """Load and train category-specific models with 4-gram context and hierarchical backoff."""
        corpora_dir = Path("data/corpora")
        if corpora_dir.exists():
            for p in corpora_dir.glob("*.txt"):
                category = p.stem
                text = p.read_text(encoding="utf-8")
                self.models[category] = SupportNGramLM(n=4, k=0.5).fit(text)

        # Main combined model
        main_corpus = Path("data/anonymized_corpus.txt")
        if not main_corpus.exists():
            milestone_1()
        self.models["general"] = SupportNGramLM(n=4, k=0.5).fit(main_corpus.read_text(encoding="utf-8"))

    def _init_default_templates(self) -> None:
        """Initialize standard support templates if empty."""
        if not self.store.get("templates"):
            default_templates = {
                "greeting": "Hello [customer], thank you for contacting retail customer support today. How can I assist you?",
                "order_shipped": "Hello [customer], your order [order_id] has been shipped via express delivery with tracking number [tracking_no].",
                "refund_issued": "We have successfully processed your full refund of [amount] to your original payment method. Please allow 3-5 business days.",
                "apology_delay": "We sincerely apologize for the delivery delay. Our fulfillment team has expedited your shipment.",
            }
            self.store.put("templates", default_templates)

    def get_model(self, category: str = "general") -> SupportNGramLM:
        return self.models.get(category, self.models.get("general"))

    def record_feedback(self, prompt: str, suggestion: str, helpful: bool, category: str = "general") -> None:
        """Store agent feedback (+1 / -1) persistently."""
        feedback_list = self.store.get("feedback_log", [])
        feedback_list.append({
            "prompt": prompt,
            "suggestion": suggestion,
            "helpful": helpful,
            "category": category,
        })
        self.store.put("feedback_log", feedback_list)

    def feedback_stats(self) -> dict[str, Any]:
        """Return analytics on suggestion helpfulness."""
        feedback_list = self.store.get("feedback_log", [])
        total = len(feedback_list)
        if total == 0:
            return {"total": 0, "helpful": 0, "unhelpful": 0, "helpfulness_rate": 0.0}
        helpful = sum(1 for item in feedback_list if item.get("helpful"))
        return {
            "total": total,
            "helpful": helpful,
            "unhelpful": total - helpful,
            "helpfulness_rate": round(helpful / total, 3),
        }

    def list_templates(self) -> dict[str, str]:
        return self.store.get("templates", {})

    def render_template(self, name: str, **kwargs) -> str:
        templates = self.list_templates()
        template = templates.get(name, "")
        for k, v in kwargs.items():
            template = template.replace(f"[{k}]", str(v)).replace(f"{{{k}}}", str(v))
        return template
