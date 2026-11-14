"""
Command-line entry point for Support Response Generator.
Usage:
  python -m app.cli hello
  python -m app.cli suggest "your order has been"
  python -m app.cli complete "thank you for contacting" --max-tokens 20
  python -m app.cli interactive
  python -m app.cli eval
  python -m app.cli template list
  python -m app.cli feedback stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import (
    SupportAssistant,
    SupportNGramLM,
    format_suggestions_table,
    generate_continuation,
    milestone_1,
    suggest_next_words,
)


def get_default_model(category: str = "general") -> SupportNGramLM:
    assistant = SupportAssistant()
    return assistant.get_model(category)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="support-response-generator",
        description="Customer support agents use this CLI tool to quickly draft responses to common queries by generating next-word suggestions based on a corpus of past support interactions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Hello command (Starter requirement)
    sub.add_parser("hello", help="check that everything is wired up")

    # 2. Suggest command
    suggest_p = sub.add_parser("suggest", help="suggest the most likely next words for a prompt")
    suggest_p.add_argument("prompt", type=str, help="partial sentence or context")
    suggest_p.add_argument("--top-k", type=int, default=3, help="number of suggestions to display")
    suggest_p.add_argument("--category", type=str, default="general", choices=["general", "shipping", "returns", "billing"], help="support domain corpus")

    # 3. Complete command
    complete_p = sub.add_parser("complete", help="autoregressively complete a support response")
    complete_p.add_argument("prompt", type=str, help="starting prompt text")
    complete_p.add_argument("--max-tokens", type=int, default=20, help="maximum tokens to generate")
    complete_p.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    complete_p.add_argument("--category", type=str, default="general", help="support domain corpus")

    # 4. Interactive agent console
    sub.add_parser("interactive", help="launch live interactive drafting console for support agents")

    # 5. Evaluation command
    eval_p = sub.add_parser("eval", help="evaluate language model perplexity across datasets")
    eval_p.add_argument("--corpus", type=str, default=None, help="path to custom corpus file")

    # 6. Anonymize utility
    anon_p = sub.add_parser("anonymize", help="scrub PII from a support log file")
    anon_p.add_argument("input_file", type=str, help="path to raw text file")
    anon_p.add_argument("output_file", type=str, help="path to save anonymized text")

    # 7. Template manager
    tmpl_p = sub.add_parser("template", help="canned support response templates")
    tmpl_p.add_argument("action", choices=["list", "render"], help="template action")
    tmpl_p.add_argument("--name", type=str, default=None, help="template name")
    tmpl_p.add_argument("--customer", type=str, default="Valued Customer", help="customer name")
    tmpl_p.add_argument("--order-id", type=str, default="#ORD-99999", help="order identifier")
    tmpl_p.add_argument("--amount", type=str, default="$50.00", help="currency amount")
    tmpl_p.add_argument("--tracking-no", type=str, default="TRK-12345678", help="shipping tracking number")

    # 8. Feedback manager
    fb_p = sub.add_parser("feedback", help="track suggestion quality feedback")
    fb_p.add_argument("action", choices=["stats", "record"], help="feedback action")
    fb_p.add_argument("--prompt", type=str, default="", help="prompt that was used")
    fb_p.add_argument("--suggestion", type=str, default="", help="suggested word")
    fb_p.add_argument("--helpful", type=str, choices=["yes", "no", "true", "false"], default="yes")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Support Response Generator is ready. Open README.md for your milestones.")

    elif args.command == "suggest":
        model = get_default_model(args.category)
        suggs = suggest_next_words(model, args.prompt, top_k=args.top_k)
        print(format_suggestions_table(args.prompt, suggs))

    elif args.command == "complete":
        model = get_default_model(args.category)
        result = generate_continuation(
            model,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n--- Generated Response ---")
        print(result)

    elif args.command == "interactive":
        assistant = SupportAssistant()
        model = assistant.get_model("general")
        print("=" * 65)
        print(" 💬 Retail Support Agent Assistant (Interactive Autocomplete)")
        print(" Type a phrase to see next-word predictions. Type ':q' to exit.")
        print("=" * 65)

        while True:
            try:
                line = input("\nagent> ").strip()
                if not line:
                    continue
                if line in {":q", "exit", "quit"}:
                    print("Exiting interactive assistant. Good luck!")
                    break

                suggs = suggest_next_words(model, line, top_k=3)
                print(format_suggestions_table(line, suggs))
                continuation = generate_continuation(model, line, max_tokens=15, temperature=0.7)
                print(f"👉 Full Auto-Complete: \"{continuation}\"")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

    elif args.command == "eval":
        from app.core import milestone_2
        res = milestone_2(args.corpus)
        print("--- Model Evaluation Metrics ---")
        print(f"Vocabulary Size: {res['vocabulary_size']}")
        print(f"Bigram Perplexity  - Train: {res['bigram']['train_ppl']:.2f} | Test: {res['bigram']['test_ppl']:.2f}")
        print(f"Trigram Perplexity - Train: {res['trigram']['train_ppl']:.2f} | Test: {res['trigram']['test_ppl']:.2f}")

    elif args.command == "anonymize":
        m1_res = milestone_1(args.input_file, args.output_file)
        print(f"✅ Anonymized corpus written to {m1_res['output_file']}")
        print(f"Stats: {m1_res['stats']}")

    elif args.command == "template":
        assistant = SupportAssistant()
        if args.action == "list":
            print("--- Canned Response Templates ---")
            for k, v in assistant.list_templates().items():
                print(f"  • {k:<18}: {v}")
        elif args.action == "render":
            if not args.name:
                print("Error: Specify template --name. Run 'template list' to view options.")
                sys.exit(1)
            rendered = assistant.render_template(
                args.name,
                customer=args.customer,
                order_id=args.order_id,
                amount=args.amount,
                tracking_no=args.tracking_no,
            )
            print(f"\nRendered Template [{args.name}]:\n{rendered}")

    elif args.command == "feedback":
        assistant = SupportAssistant()
        if args.action == "stats":
            stats = assistant.feedback_stats()
            print("--- Agent Feedback Analytics ---")
            print(f"Total Feedback Logged : {stats['total']}")
            print(f"Helpful Suggestions   : {stats['helpful']}")
            print(f"Unhelpful Suggestions : {stats['unhelpful']}")
            print(f"Helpfulness Rate      : {stats['helpfulness_rate'] * 100:.1f}%")
        elif args.action == "record":
            is_helpful = args.helpful.lower() in {"yes", "true"}
            assistant.record_feedback(args.prompt, args.suggestion, is_helpful)
            print("✅ Feedback recorded successfully.")


if __name__ == "__main__":
    main()
