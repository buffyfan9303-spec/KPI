"""4단계 뉴스 게이트 (PDF 8쪽). 한국어 FinBERT로 제목을 판정해 치명적 악재면 REJECT."""
from functools import lru_cache

MODEL = "snunlp/KR-FinBert-SC"  # 라이선스 표기 없음 → 개인 연구용
# FinBERT만으로는 "다시 들썩" 같은 호재 제목도 부정으로 잡아서(2026-09-25 실측) 치명적 사건 단어와 함께 볼 때만 보류한다.
# 조정 가능: 단어를 더하거나 빼면 게이트 민감도가 바뀐다.
CRITICAL = ("횡령", "배임", "상장폐지", "상장적격성", "거래정지", "매매거래정지", "감사의견", "의견거절",
            "부도", "회생절차", "파산", "불성실공시", "분식", "압수수색", "기소", "영업정지", "리콜")


@lru_cache(maxsize=1)
def _model():
    from transformers import pipeline  # 무거운 import는 첫 호출 때만
    return pipeline("text-classification", model=MODEL)


def classify(titles: list[str]) -> list[dict]:
    if not titles:
        return []
    return [{"label": r["label"], "score": round(float(r["score"]), 3)}
            for r in _model()(titles, truncation=True)]


def critical(item: dict) -> bool:
    """치명적 사건 단어가 있고 FinBERT가 긍정으로 보지 않은 제목."""
    return item.get("label") != "positive" and any(k in item["title"] for k in CRITICAL)


def gate(items: list[dict]) -> str:
    return "REJECT" if any(critical(i) for i in items) else "EXECUTE"
