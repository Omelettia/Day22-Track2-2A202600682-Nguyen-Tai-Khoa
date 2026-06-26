"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
import types
import os
import time
import inspect
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAGAS 0.4.x imports the legacy VertexAI chat model path at module import time.
# LangChain Community 0.4 removed that module, and this lab does not use VertexAI,
# so a minimal shim keeps RAGAS import-compatible with the modern LangChain stack.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - compatibility shim only
        pass

    vertexai_shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_shim

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


FREE_TIER_MODE = os.getenv("RAGAS_FREE_TIER_MODE", "true").lower() in ("1", "true", "yes")
DEFAULT_SAMPLE_LIMIT = "5" if FREE_TIER_MODE and config.PROVIDER == "gemini" else "0"
RAGAS_SAMPLE_LIMIT = int(os.getenv("RAGAS_SAMPLE_LIMIT", DEFAULT_SAMPLE_LIMIT) or "0")
RAGAS_SLEEP_SECONDS = float(os.getenv("RAGAS_SLEEP_SECONDS", "8" if FREE_TIER_MODE else "0") or "0")
RAGAS_EVAL_TIMEOUT = int(os.getenv("RAGAS_EVAL_TIMEOUT", "180") or "180")
RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "6" if FREE_TIER_MODE else "3") or "3")
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "1" if FREE_TIER_MODE else "4") or "1")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn trong 2-4 câu, rõ ràng và chỉ dựa "
    "trên context được cung cấp. Nếu context không chứa thông tin cần thiết, hãy nói "
    "rằng bạn không tìm thấy thông tin.\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Hãy đọc kỹ context, xác định các dữ kiện "
    "liên quan, rồi trả lời có cấu trúc trong 3-5 câu với giọng chuyên nghiệp. Không "
    "suy đoán ngoài context; nếu thiếu dữ kiện, nêu rõ rằng context không đủ thông tin.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=2000, chunk_overlap=100)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    chain = prompt | llm | StrOutputParser()
    last_error = None
    for attempt in range(1, RAGAS_MAX_RETRIES + 1):
        try:
            answer = chain.invoke({
                "context":  ctx_str,
                "question": question,
            })
            break
        except Exception as exc:
            last_error = exc
            wait = min(RAGAS_SLEEP_SECONDS * attempt, 60)
            print(f"    ⚠️ Lỗi LLM lần {attempt}/{RAGAS_MAX_RETRIES}: {exc}. Chờ {wait:.0f}s...")
            time.sleep(wait)
    else:
        raise last_error

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    qa_pairs = QA_PAIRS[:RAGAS_SAMPLE_LIMIT] if RAGAS_SAMPLE_LIMIT > 0 else QA_PAIRS
    total = len(qa_pairs)

    print(f"\n🚀 Đang chạy {total} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(qa_pairs, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/{total}] {qa['question'][:60]}")
        if RAGAS_SLEEP_SECONDS > 0 and i < total:
            time.sleep(RAGAS_SLEEP_SECONDS)

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    evaluate_kwargs = {
        "dataset": dataset,
        "metrics": [faithfulness, answer_relevancy, context_recall, context_precision],
        "llm": llm_eval,
        "embeddings": emb_eval,
    }

    eval_params = inspect.signature(evaluate).parameters
    if "batch_size" in eval_params:
        evaluate_kwargs["batch_size"] = 1

    if "run_config" in eval_params:
        try:
            from ragas.run_config import RunConfig

            run_config_values = {
                "timeout": RAGAS_EVAL_TIMEOUT,
                "max_retries": RAGAS_MAX_RETRIES,
                "max_wait": 60,
                "max_workers": RAGAS_MAX_WORKERS,
            }
            run_config_params = inspect.signature(RunConfig).parameters
            evaluate_kwargs["run_config"] = RunConfig(**{
                key: value
                for key, value in run_config_values.items()
                if key in run_config_params
            })
        except Exception as exc:
            print(f"  ⚠️ Không tạo được RAGAS RunConfig, chạy cấu hình mặc định: {exc}")

    result = evaluate(**evaluate_kwargs)

    # Tính mean score cho mỗi metric. RAGAS trả về NaN nếu một job bị timeout/lỗi,
    # nên cần bỏ qua NaN thay vì để nó làm hỏng toàn bộ mean.
    scores = {}
    diagnostics = {}
    for key in RAGAS_METRICS:
        raw = list(result[key])
        values = [
            float(v)
            for v in raw
            if v is not None and not np.isnan(float(v))
        ]
        scores[key] = float(np.mean(values)) if values else 0.0
        diagnostics[key] = {
            "valid": len(values),
            "failed": len(raw) - len(values),
            "total": len(raw),
        }

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        diag = diagnostics[k]
        print(f"  {k:30s}: {v:.4f}{star}  ({diag['valid']}/{diag['total']} valid)")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in RAGAS_METRICS:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
