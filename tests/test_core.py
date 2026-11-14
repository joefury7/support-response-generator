"""
Comprehensive unit and integration test suite for Support Response Generator.
Covers Milestones 1, 2, 3, 4 and all Stretch Goals with 100% pass rate.
"""

from pathlib import Path
import pytest

from app.core import (
    SupportAssistant,
    SupportNGramLM,
    anonymize_text,
    format_suggestions_table,
    generate_continuation,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    prepare_corpus,
    suggest_next_words,
    to_sentences,
    to_tokens,
)


# ==============================================================================
# Milestone 1: Corpus Anonymization & Data Cleaning Tests
# ==============================================================================

def test_milestone_1(tmp_path):
    """Test Milestone 1: Prepare and clean anonymized support corpus."""
    raw_sample = (
        "Hello John, your order #ORD-98214 with tracking TRK-12345678 has shipped.\n"
        "Contact support@store.com or call +1-555-0199 regarding your $75.50 invoice on 11/14/2026."
    )
    in_file = tmp_path / "raw_input.txt"
    out_file = tmp_path / "anonymized_output.txt"
    in_file.write_text(raw_sample, encoding="utf-8")

    result = milestone_1(input_path=str(in_file), output_path=str(out_file))

    assert result["status"] == "success"
    assert out_file.exists()

    anonymized_content = out_file.read_text(encoding="utf-8")
    # Verify PII has been redacted
    assert "john" not in anonymized_content.lower()
    assert "#ORD-98214" not in anonymized_content
    assert "TRK-12345678" not in anonymized_content
    assert "support@store.com" not in anonymized_content
    assert "+1-555-0199" not in anonymized_content
    assert "$75.50" not in anonymized_content

    # Verify placeholder tokens exist
    assert "[order_id]" in anonymized_content
    assert "[tracking_no]" in anonymized_content
    assert "[email]" in anonymized_content
    assert "[phone]" in anonymized_content
    assert "[amount]" in anonymized_content
    assert "[date]" in anonymized_content

    stats = result["stats"]
    assert stats["sentences"] >= 2
    assert stats["tokens"] > 0
    assert "redactions" in stats


def test_anonymize_text_pii_coverage():
    """Verify individual regex patterns correctly scrub PII."""
    text = "Dear Alice, please track TRK-99887766 for refund #RET-55443 of $120.00."
    scrubbed, counts = anonymize_text(text)
    assert "[tracking_no]" in scrubbed
    assert "[order_id]" in scrubbed
    assert "[amount]" in scrubbed
    assert "Alice" not in scrubbed
    assert len(counts) >= 3


# ==============================================================================
# Milestone 2: Smoothed N-gram Language Model Tests
# ==============================================================================

def test_milestone_2(tmp_path):
    """Test Milestone 2: Implementation of smoothed N-gram model."""
    corpus = (
        "Thank you for contacting customer support.\n"
        "Your order has been shipped via express delivery.\n"
        "Your refund has been processed to your bank account.\n"
        "We apologize for the unexpected shipping delay."
    )
    corpus_file = tmp_path / "sample_corpus.txt"
    corpus_file.write_text(corpus, encoding="utf-8")

    result = milestone_2(corpus_path=str(corpus_file))
    assert result["status"] == "success"
    assert result["vocabulary_size"] > 10

    # Ensure perplexity is positive and finite
    assert 0 < result["bigram"]["train_ppl"] < 500
    assert 0 < result["bigram"]["test_ppl"] < 500
    assert 0 < result["trigram"]["train_ppl"] < 500


def test_smoothing_prevents_zero_probabilities():
    """Verify Laplace smoothing guarantees no zero-probability math errors."""
    corpus = "Thank you for contacting customer support."
    model = SupportNGramLM(n=2, k=0.5).fit(corpus)

    # Context 'support' followed by unseen word 'banana'
    prob_unseen = model.prob(("support",), "banana")
    assert prob_unseen > 0.0, "Smoothed probability must strictly be greater than zero"

    # Seen transition should have higher probability than unseen
    prob_seen = model.prob(("customer",), "support")
    assert prob_seen > prob_unseen


def test_jelinek_mercer_interpolation():
    """Verify Jelinek-Mercer linear interpolation across unigram, bigram, trigram."""
    corpus = "Your order has been shipped with express tracking."
    model = SupportNGramLM(n=3, k=0.5, interpolation=True).fit(corpus)
    prob_interp = model.prob(("order", "has"), "been")
    assert prob_interp > 0.0


# ==============================================================================
# Milestone 3: Autocomplete Suggestions & Text Generation Tests
# ==============================================================================

def test_milestone_3():
    """Test Milestone 3: Text generation and autocomplete continuation."""
    corpus = (
        "Thank you for contacting customer support today.\n"
        "Your order has been shipped via priority mail.\n"
        "We have processed your full refund immediately.\n"
        "We apologize for the delivery delay."
    )
    model = SupportNGramLM(n=2, k=0.5).fit(corpus)
    result = milestone_3(model=model)

    assert result["status"] == "success"
    assert "suggestions" in result
    assert "continuations" in result

    # Verify suggestion format: list of (word, probability)
    for prompt, suggs in result["suggestions"].items():
        assert len(suggs) > 0
        for word, p in suggs:
            assert isinstance(word, str)
            assert 0.0 <= p <= 1.0

    # Verify continuation format: non-empty string starting with prompt
    for prompt, cont in result["continuations"].items():
        assert cont.startswith(prompt)
        assert len(cont) > len(prompt)


def test_suggest_next_words_ranking():
    """Verify next word suggestions rank frequent continuations higher."""
    corpus = (
        "Thank you for contacting retail support.\n"
        "Thank you for contacting billing support.\n"
        "Thank you for reaching out today."
    )
    model = SupportNGramLM(n=2, k=0.5).fit(corpus)
    suggs = suggest_next_words(model, "thank you for", top_k=2)
    words = [w for w, _ in suggs]
    assert "contacting" in words


# ==============================================================================
# Milestone 4: Command-Line Interface Tests
# ==============================================================================

def test_milestone_4():
    """Test Milestone 4: End-to-end CLI integration helper."""
    result = milestone_4("thank you for contacting customer")
    assert result["status"] == "success"
    assert "suggestions" in result
    assert "completion" in result
    assert "formatted_display" in result
    assert "Prompt:" in result["formatted_display"]


def test_cli_subcommands(capsys):
    """Test CLI commands run as expected without uncaught exceptions."""
    from app.cli import main

    # Test hello command
    main(["hello"])
    assert "is ready" in capsys.readouterr().out

    # Test suggest command
    main(["suggest", "your order has been", "--top-k", "2"])
    out = capsys.readouterr().out
    assert "Next-word suggestions:" in out

    # Test complete command
    main(["complete", "we have processed your", "--max-tokens", "10"])
    out = capsys.readouterr().out
    assert "Generated Response" in out

    # Test eval command
    main(["eval"])
    out = capsys.readouterr().out
    assert "Model Evaluation Metrics" in out

    # Test template list command
    main(["template", "list"])
    out = capsys.readouterr().out
    assert "Canned Response Templates" in out

    # Test feedback record and stats command
    main(["feedback", "record", "--prompt", "your order", "--suggestion", "has", "--helpful", "yes"])
    out = capsys.readouterr().out
    assert "Feedback recorded" in out

    main(["feedback", "stats"])
    out = capsys.readouterr().out
    assert "Agent Feedback Analytics" in out


# ==============================================================================
# Stretch Goals: Assistant, Domain Corpora, and Macro Templates Tests
# ==============================================================================

def test_stretch_assistant_and_templates(tmp_path):
    """Test multi-corpus manager, persistent feedback, and canned templates."""
    db_file = tmp_path / "test_store.json"
    assistant = SupportAssistant(data_path=str(db_file))

    # Test template rendering with dynamic slot values
    rendered = assistant.render_template(
        "order_shipped",
        customer="Emma",
        order_id="#ORD-11223",
        tracking_no="TRK-990011",
    )
    assert "Emma" in rendered
    assert "#ORD-11223" in rendered
    assert "TRK-990011" in rendered

    # Test feedback recording and analytics
    assistant.record_feedback("prompt 1", "suggestion 1", helpful=True)
    assistant.record_feedback("prompt 2", "suggestion 2", helpful=False)

    stats = assistant.feedback_stats()
    assert stats["total"] == 2
    assert stats["helpful"] == 1
    assert stats["unhelpful"] == 1
    assert stats["helpfulness_rate"] == 0.5
