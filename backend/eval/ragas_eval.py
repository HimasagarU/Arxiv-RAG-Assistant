"""
ragas_eval.py — Run reference-free RAGAS metrics against the RAG pipeline.

Evaluates faithfulness, response relevancy, and context precision for
each question in ragas_queries.jsonl by calling the retriever + LLM
directly (no network round-trip).

Usage:
    conda run -n pytorch python eval/ragas_eval.py
    conda run -n pytorch python eval/ragas_eval.py --api-url https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space
    conda run -n pytorch python eval/ragas_eval.py --limit 5       # quick smoke test
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_PATH = EVAL_DIR / "ragas_queries.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


# ---------------------------------------------------------------------------
# Pipeline runner — calls retriever + LLM locally OR via deployed API
# ---------------------------------------------------------------------------

def _run_query_local(question: str, retriever, top_n: int = 5) -> dict:
    """Run the full retrieval + generation pipeline locally."""
    from api.retrieval import classify_query_intent
    from api.app import build_prompt, generate_answer

    intent = classify_query_intent(question)
    t0 = time.time()

    result = retriever.retrieve(question, top_n=top_n, intent=intent)
    passages = result["passages"]
    trace = result["trace"]

    compressed_context = retriever.compress_context(question, passages, intent=intent)
    prompt = build_prompt(question, compressed_context, passages, intent=intent)
    answer = generate_answer(prompt, intent=intent)
    total_ms = round((time.time() - t0) * 1000, 1)

    contexts = [p["chunk_text"] for p in passages if p.get("chunk_text")]

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "contexts": contexts,
        "trace": trace,
        "latency_ms": total_ms,
        "chunk_ids": [p["chunk_id"] for p in passages],
    }


def _run_query_api(question: str, api_url: str, top_k: int = 5) -> dict:
    """Run the full pipeline via deployed API."""
    import requests as req

    t0 = time.time()
    resp = req.post(
        f"{api_url.rstrip('/')}/query",
        json={"query": question, "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    total_ms = round((time.time() - t0) * 1000, 1)

    contexts = [s["chunk_text"] for s in data.get("sources", []) if s.get("chunk_text")]

    return {
        "question": question,
        "intent": data.get("retrieval_trace", {}).get("intent", "discovery"),
        "answer": data.get("answer", ""),
        "contexts": contexts,
        "trace": data.get("retrieval_trace", {}),
        "latency_ms": total_ms,
        "chunk_ids": [s["chunk_id"] for s in data.get("sources", [])],
    }


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

async def evaluate_with_ragas(records: list[dict], evaluator_llm, output_path: str = None, existing_scored: list = None) -> list[dict]:
    """Score each record with RAGAS reference-free metrics.
    
    Note: ResponseRelevancy uses n>1 generation which Groq does not support.
    We attempt it but gracefully fall back if the LLM provider rejects it.
    """
    from ragas import SingleTurnSample
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    context_precision_metric = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    # ResponseRelevancy needs n>1 generation — try once, disable if provider rejects
    relevancy_metric = ResponseRelevancy(llm=evaluator_llm)
    relevancy_supported = True

    scored = []
    for i, rec in enumerate(records):
        log.info(f"  [{i+1}/{len(records)}] Scoring: {rec['question'][:60]}...")

        sample = SingleTurnSample(
            user_input=rec["question"],
            response=rec["answer"],
            retrieved_contexts=rec["contexts"],
        )

        try:
            faith_score = await faithfulness_metric.single_turn_ascore(sample)
        except Exception as e:
            log.warning(f"  faithfulness failed: {e}")
            faith_score = None

        relev_score = None
        if relevancy_supported:
            try:
                relev_score = await relevancy_metric.single_turn_ascore(sample)
            except Exception as e:
                if "n" in str(e).lower() and ("must be" in str(e).lower() or "at most" in str(e).lower()):
                    log.warning(f"  response_relevancy disabled: LLM provider does not support n>1 generation")
                    relevancy_supported = False
                else:
                    log.warning(f"  response_relevancy failed: {e}")

        try:
            ctx_prec_score = await context_precision_metric.single_turn_ascore(sample)
        except Exception as e:
            log.warning(f"  context_precision failed: {e}")
            ctx_prec_score = None

        rec["ragas"] = {
            "faithfulness": faith_score,
            "response_relevancy": relev_score,
            "context_precision": ctx_prec_score,
        }
        scored.append(rec)
        
        # Incremental save so we never lose progress if the script crashes
        if output_path and existing_scored is not None:
            combined_scored = [r for r in existing_scored if r["question"] not in [n["question"] for n in scored]]
            combined_scored.extend(scored)
            
            # Make a copy and strip trace for saving
            save_data = []
            for item in combined_scored:
                item_copy = dict(item)
                item_copy.pop("trace", None)
                save_data.append(item_copy)
                
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"scored": save_data, "total_queries": len(save_data)}, f, indent=2, default=str)

    if not relevancy_supported:
        log.info("  ℹ️  response_relevancy was skipped (Groq does not support n>1).")
        log.info("     To enable it, use --llm-model with an OpenAI-compatible provider.")

    return scored


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(scored: list[dict]):
    """Print per-intent and aggregate summaries."""
    from collections import defaultdict

    by_intent = defaultdict(list)
    for rec in scored:
        by_intent[rec["intent"]].append(rec)

    print("\n" + "=" * 72)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 72)

    all_faith, all_relev, all_ctx = [], [], []

    for intent in sorted(by_intent):
        recs = by_intent[intent]
        faith_vals = [r["ragas"]["faithfulness"] for r in recs if r["ragas"]["faithfulness"] is not None]
        relev_vals = [r["ragas"]["response_relevancy"] for r in recs if r["ragas"]["response_relevancy"] is not None]
        ctx_vals = [r["ragas"]["context_precision"] for r in recs if r["ragas"]["context_precision"] is not None]
        lat_vals = [r["latency_ms"] for r in recs]

        all_faith.extend(faith_vals)
        all_relev.extend(relev_vals)
        all_ctx.extend(ctx_vals)

        def _avg(vals):
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\n  Intent: {intent} ({len(recs)} queries)")
        print(f"    faithfulness       : {_avg(faith_vals):.4f}  (n={len(faith_vals)})")
        print(f"    response_relevancy : {_avg(relev_vals):.4f}  (n={len(relev_vals)})")
        print(f"    context_precision  : {_avg(ctx_vals):.4f}  (n={len(ctx_vals)})")
        print(f"    avg_latency_ms     : {_avg(lat_vals):.1f}")

    def _avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n  ── AGGREGATE ──")
    print(f"    faithfulness       : {_avg(all_faith):.4f}  (n={len(all_faith)})")
    print(f"    response_relevancy : {_avg(all_relev):.4f}  (n={len(all_relev)})")
    print(f"    context_precision  : {_avg(all_ctx):.4f}  (n={len(all_ctx)})")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for ArXiv RAG")
    parser.add_argument("--queries", type=str, default=str(QUERIES_PATH))
    parser.add_argument("--api-url", type=str, default=None,
                        help="If set, run queries against deployed API instead of local retriever.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of queries (for quick smoke tests).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for per-query results.")
    parser.add_argument("--llm-model", type=str, default="llama-3.3-70b-versatile",
                        help="LLM model for RAGAS evaluation judge.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing results and start from scratch.")
    args = parser.parse_args()

    # Load questions
    if not os.path.exists(args.queries):
        log.error(f"Queries file not found: {args.queries}. Run `python eval/ragas_dataset.py` first.")
        return

    questions = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[:args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(RESULTS_DIR / "ragas_results.json")

    # Load existing results for resume capability
    existing_scored = []
    fully_evaluated_questions = set()
    answered_records = {}
    
    if not args.restart and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_scored = data.get("scored", [])
                for rec in existing_scored:
                    answered_records[rec["question"]] = rec
                    ragas_scores = rec.get("ragas", {})
                    # If it has a faithfulness score, we consider it successfully evaluated
                    if ragas_scores.get("faithfulness") is not None:
                        fully_evaluated_questions.add(rec["question"])
        except Exception as e:
            log.warning(f"Could not load existing results: {e}")

    questions_needing_answers = [q for q in questions if q["question"] not in answered_records]
    records_needing_eval = [answered_records[q["question"]] for q in questions if q["question"] in answered_records and q["question"] not in fully_evaluated_questions]

    log.info(f"Loaded {len(questions)} total questions.")
    if answered_records:
        log.info(f"  {len(fully_evaluated_questions)} fully evaluated. {len(records_needing_eval)} have answers but need evaluation. {len(questions_needing_answers)} need answers.")

    if not questions_needing_answers and not records_needing_eval:
        log.info("All questions have already been evaluated! Use --restart to start over.")
        print_summary(existing_scored)
        return

    # ── Step 1: Run queries through the pipeline ──
    log.info("Step 1: Running queries through RAG pipeline...")
    
    def _save_intermediate(new_rec):
        combined = [r for r in existing_scored if r["question"] != new_rec["question"]]
        combined.append(new_rec)
        existing_scored.clear()
        existing_scored.extend(combined)
        
        save_data = []
        for item in existing_scored:
            item_copy = dict(item)
            item_copy.pop("trace", None)
            save_data.append(item_copy)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"scored": save_data, "total_queries": len(save_data)}, f, indent=2, default=str)

    if questions_needing_answers:
        if args.api_url:
            log.info(f"  Using deployed API: {args.api_url}")
            for i, q in enumerate(questions_needing_answers):
                log.info(f"  [{i+1}/{len(questions_needing_answers)}] {q['question'][:60]}...")
                try:
                    rec = _run_query_api(q["question"], args.api_url)
                    rec["intent"] = q.get("intent", rec.get("intent", "discovery"))
                    records_needing_eval.append(rec)
                    _save_intermediate(rec)
                except Exception as e:
                    log.error(f"  API call failed: {e}")
        else:
            log.info("  Using local retriever...")
            from api.retrieval import HybridRetriever
            retriever = HybridRetriever()

            for i, q in enumerate(questions_needing_answers):
                log.info(f"  [{i+1}/{len(questions_needing_answers)}] {q['question'][:60]}...")
                try:
                    rec = _run_query_local(q["question"], retriever)
                    rec["intent"] = q.get("intent", rec.get("intent", "discovery"))
                    records_needing_eval.append(rec)
                    _save_intermediate(rec)
                except Exception as e:
                    log.error(f"  Pipeline failed: {e}")

    if not records_needing_eval:
        log.error("No new records were successfully generated to evaluate.")
        if existing_scored:
            print_summary(existing_scored)
        return

    # ── Step 2: Run RAGAS metrics ──
    log.info(f"Step 2: Running RAGAS metrics with judge LLM: {args.llm_model}...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        log.error("GROQ_API_KEY is required for RAGAS evaluation. Set it in .env.")
        return

    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=args.llm_model, api_key=groq_api_key, temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    new_scored = asyncio.run(evaluate_with_ragas(records_needing_eval, evaluator_llm, output_path, existing_scored))

    # Strip non-serializable trace data for the final combined list
    for rec in new_scored:
        rec.pop("trace", None)

    # ── Step 3: Combine and Save (Final) ──
    combined_scored = [r for r in existing_scored if r["question"] not in [n["question"] for n in new_scored]]
    combined_scored.extend(new_scored)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"scored": combined_scored, "total_queries": len(combined_scored)}, f, indent=2, default=str)

    log.info(f"Final results safely stored → {output_path}")

    # ── Step 4: Report ──
    print_summary(combined_scored)


if __name__ == "__main__":
    main()
