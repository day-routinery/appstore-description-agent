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
    app_info: AppInfo | dict[str, Any],
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
# CLI 데모
# ---------------------------------------------------------------------------

def main():
    # ── 앱 정보 ──────────────────────────────────────────────────────────
    app = AppInfo(
        name="FocusFlow",
        category="Productivity",
        target_audience="Professionals and students who struggle with distractions",
        core_feature="AI-powered focus timer combined with habit tracking and smart break reminders",
        additional_context="Won 'Best Productivity App 2024' by AppAdvice. Over 500,000 downloads.",
    )

    # ── 포함할 요소 및 분량 지정 ─────────────────────────────────────────
    elements = [
        Element(
            name="authority",
            description="Establish credibility — mention award or user base",
            sentences=2,
        ),
        Element(
            name="pain_point",
            description="Describe the core problem the target audience faces daily",
            sentences=3,
        ),
        Element(
            name="solution",
            description="Explain how the app solves that problem uniquely",
            sentences=4,
        ),
        Element(
            name="features",
            description="List the top 5 key features",
            sentences=5,
            format="bullet",
        ),
        Element(
            name="cta",
            description="Strong call-to-action encouraging download",
            sentences=1,
        ),
    ]

    # ── 생성 ─────────────────────────────────────────────────────────────
    descriptions = generate_descriptions(
        app_info=app,
        elements=elements,
        max_chars=4000,
    )

    # ── 결과 저장 (선택) ─────────────────────────────────────────────────
    output_path = "descriptions_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== ENGLISH ===\n\n")
        f.write(descriptions["en"])
        f.write("\n\n\n=== JAPANESE ===\n\n")
        f.write(descriptions["ja"])

    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
