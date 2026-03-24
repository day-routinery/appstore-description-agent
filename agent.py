"""
App Store Description Agent
----------------------------
Apple App Store용 description을 영어와 일본어로 자동 생성하는 에이전트.

사용법:
    python agent.py

또는 코드에서 직접:
    from agent import generate_descriptions

    descriptions = generate_descriptions(
        app_info={
            "name": "FocusFlow",
            "category": "Productivity",
            "target_audience": "직장인, 학생",
            "core_feature": "AI 기반 집중 타이머 + 습관 추적",
        },
        elements=[
            Element("authority",   description="개발사 신뢰도나 수상 이력 등 권위 확립",    sentences=2),
            Element("pain_point",  description="사용자가 겪는 핵심 문제",                    sentences=3),
            Element("solution",    description="앱이 제공하는 해결책",                       sentences=4),
            Element("features",    description="주요 기능 목록 (bullet 형식)",               sentences=5),
            Element("cta",         description="다운로드 유도 문구",                         sentences=1),
        ],
        max_chars=4000,  # App Store 최대 4000자
    )

    print(descriptions["en"])
    print(descriptions["ja"])
"""

import sys
import os
import anthropic
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------

@dataclass
class Element:
    """description에 포함할 요소 하나."""
    name: str                         # 예: "pain_point", "solution"
    description: str                  # 이 요소가 무엇을 담아야 하는지 설명
    sentences: int = 2                # 목표 문장 수
    format: str = "paragraph"         # "paragraph" | "bullet"


@dataclass
class AppInfo:
    """앱 기본 정보."""
    name: str
    category: str
    target_audience: str
    core_feature: str
    additional_context: str = ""      # 있으면 전달할 추가 정보 (수상 이력, 경쟁 앱 등)


# ---------------------------------------------------------------------------
# 프롬프트 빌더
# ---------------------------------------------------------------------------

def _build_elements_spec(elements: list[Element]) -> str:
    lines = []
    for i, el in enumerate(elements, 1):
        fmt_note = " (use bullet points, one per line)" if el.format == "bullet" else ""
        lines.append(
            f"{i}. [{el.name.upper()}] — {el.description}\n"
            f"   Target length: ~{el.sentences} sentence(s){fmt_note}"
        )
    return "\n".join(lines)


def _build_prompt(app_info: AppInfo, elements: list[Element], max_chars: int, lang: str) -> str:
    lang_label = {"en": "English", "ja": "Japanese"}.get(lang, lang)
    lang_note = (
        "Write in natural, fluent English suitable for an English-speaking audience."
        if lang == "en"
        else "Write in natural, fluent Japanese (日本語) suitable for the Japanese App Store. "
             "Use polite but engaging tone (です・ます調 or a mix). "
             "Avoid direct translation feel — write as a native Japanese copywriter would."
    )

    elements_spec = _build_elements_spec(elements)

    return f"""You are an expert App Store copywriter.
Write an Apple App Store description in {lang_label} for the following app.

## App Information
- Name: {app_info.name}
- Category: {app_info.category}
- Target audience: {app_info.target_audience}
- Core feature: {app_info.core_feature}
{f"- Additional context: {app_info.additional_context}" if app_info.additional_context else ""}

## Required Elements (in order)
Include ALL of the following elements in this exact order:

{elements_spec}

## Constraints
- Total length: under {max_chars} characters
- {lang_note}
- Do NOT include section headers or labels (e.g., don't write "[AUTHORITY]" in the output)
- The text should flow naturally as a single cohesive description
- Paragraphs should be separated by a blank line

Output ONLY the final description text. No preamble, no explanation.
"""


# ---------------------------------------------------------------------------
# 메인 생성 함수
# ---------------------------------------------------------------------------

def generate_descriptions(
    app_info: "AppInfo | dict[str, Any]",
    elements: list[Element],
    max_chars: int = 4000,
    model: str = "claude-opus-4-6",
    verbose: bool = True,
) -> dict[str, str]:
    """
    영어 + 일본어 App Store description을 생성하고 반환합니다.

    Returns:
        {"en": "...", "ja": "..."}
    """
    if isinstance(app_info, dict):
        app_info = AppInfo(**app_info)

    client = anthropic.Anthropic()
    results: dict[str, str] = {}

    for lang, lang_label in [("en", "English"), ("ja", "Japanese")]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating {lang_label} description...")
            print(f"{'='*60}\n")

        prompt = _build_prompt(app_info, elements, max_chars, lang)

        full_text = ""
        with client.messages.stream(
            model=model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    chunk = event.delta.text
                    full_text += chunk
                    if verbose:
                        print(chunk, end="", flush=True)

        results[lang] = full_text.strip()

        if verbose:
            char_count = len(results[lang])
            print(f"\n\n[{lang_label}: {char_count} chars / {max_chars} max]")

    return results


# ---------------------------------------------------------------------------
# 기본 요소 목록
# ---------------------------------------------------------------------------

DEFAULT_ELEMENTS = [
    ("authority",  "개발사 신뢰도, 수상 이력, 다운로드 수 등 권위 확립"),
    ("pain_point", "타겟 사용자가 겪는 핵심 문제"),
    ("solution",   "앱이 그 문제를 해결하는 방식"),
    ("features",   "주요 기능 목록"),
    ("cta",        "다운로드 유도 문구"),
]


# ---------------------------------------------------------------------------
# 인터랙티브 입력
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    value = input(f"{prompt}{hint}: ").strip()
    return value if value else default


def collect_app_info() -> AppInfo:
    print("\n" + "="*60)
    print("  앱 정보를 입력해 주세요")
    print("="*60 + "\n")

    name = ask("앱 이름")
    category = ask("카테고리 (예: Productivity, Health, Finance)")
    target_audience = ask("타겟 사용자 (예: 직장인, 학생, 부모)")
    core_feature = ask("핵심 기능 한 줄 설명")
    additional_context = ask("추가 정보 (수상 이력, 다운로드 수 등, 없으면 Enter)", default="")

    return AppInfo(
        name=name,
        category=category,
        target_audience=target_audience,
        core_feature=core_feature,
        additional_context=additional_context,
    )


def collect_elements() -> list[Element]:
    print("\n" + "="*60)
    print("  포함할 요소를 설정해 주세요")
    print("  (Enter = 기본값 사용, 0 입력 = 해당 요소 제외)")
    print("="*60 + "\n")

    elements = []
    for name, desc in DEFAULT_ELEMENTS:
        print(f"[{name.upper()}] {desc}")
        sentences_input = ask("  문장 수", default="2")
        if sentences_input == "0":
            print("  → 제외됨\n")
            continue
        try:
            sentences = int(sentences_input)
        except ValueError:
            sentences = 2

        fmt = "paragraph"
        if name == "features":
            fmt_input = ask("  형식 (1=bullet, 2=paragraph)", default="1")
            fmt = "bullet" if fmt_input != "2" else "paragraph"

        elements.append(Element(name=name, description=desc, sentences=sentences, format=fmt))
        print()

    return elements


def collect_max_chars() -> int:
    print("="*60)
    val = ask("최대 글자 수 (App Store 최대 4000)", default="4000")
    try:
        return int(val)
    except ValueError:
        return 4000


# ---------------------------------------------------------------------------
# CLI 메인
# ---------------------------------------------------------------------------

def main():
    print("\n🍎 App Store Description Generator")

    app = collect_app_info()
    elements = collect_elements()

    if not elements:
        print("요소를 하나 이상 선택해 주세요.")
        return

    max_chars = collect_max_chars()

    descriptions = generate_descriptions(
        app_info=app,
        elements=elements,
        max_chars=max_chars,
    )

    output_path = "descriptions_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== ENGLISH ===\n\n")
        f.write(descriptions["en"])
        f.write("\n\n\n=== JAPANESE ===\n\n")
        f.write(descriptions["ja"])

    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
