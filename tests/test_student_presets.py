"""
tests/test_student_presets.py — Unit tests for Student Q&A Preset Engine.
"""

from __future__ import annotations

import pytest

from core.student_presets import (
    STUDENT_PRESETS,
    find_preset_answer,
    get_presets_by_category,
    normalize_query,
)
from core.llm import GroqClientWrapper
from config import LLMConfig


class TestStudentPresetsEngine:
    def test_presets_knowledge_bank_populated(self):
        assert len(STUDENT_PRESETS) >= 15
        for preset in STUDENT_PRESETS:
            assert "id" in preset
            assert "category" in preset
            assert "question" in preset
            assert "answer" in preset
            assert len(preset["triggers"]) > 0

    def test_normalize_query(self):
        raw = "  What's Photosynthesis, Please?  "
        normalized = normalize_query(raw)
        assert "photosynthesis" in normalized
        assert "?" not in normalized
        assert "," not in normalized

    def test_exact_trigger_matches(self):
        # Science
        ans1 = find_preset_answer("what is photosynthesis")
        assert ans1 is not None
        assert "green plants" in ans1.lower()

        # Math
        ans2 = find_preset_answer("explain pythagoras theorem")
        assert ans2 is not None
        assert "squared" in ans2.lower()

        # Computer Science
        ans3 = find_preset_answer("what is python")
        assert ans3 is not None
        assert "programming language" in ans3.lower()

        # Study Skills
        ans4 = find_preset_answer("how to study effectively")
        assert ans4 is not None
        assert "active recall" in ans4.lower()

        # Creator Info
        ans5 = find_preset_answer("who created you")
        assert ans5 is not None
        assert "al irshad public school" in ans5.lower()

    def test_fuzzy_variation_matches(self):
        # User adds extra words or variations
        ans = find_preset_answer("hey nyra can you tell me what is gravity please")
        assert ans is not None
        assert "attraction" in ans.lower()

        ans_pi = find_preset_answer("what is the value of pi")
        assert ans_pi is not None
        assert "3.14159" in ans_pi

    def test_unmatched_query_returns_none(self):
        # Non-preset query should return None to trigger LLM fallback
        ans = find_preset_answer("what is the current temperature in Tokyo?")
        assert ans is None

    def test_zain_student_performance_query(self):
        # Parent query asking about previous month performance of son Zain
        ans = find_preset_answer("what is the previous month performance of my son zain")
        assert ans is not None
        assert "95%" in ans
        assert "Science" in ans
        assert "Robotics" in ans
        assert "essay writing" in ans or "time allocation" in ans

        # Parent query asking about strengths and weaknesses
        ans_sw = find_preset_answer("what are zain's strengths and weaknesses")
        assert ans_sw is not None
        assert "analytical problem-solving" in ans_sw.lower()
        assert "time management" in ans_sw.lower()

    def test_get_student_record_summary(self):
        from core.student_presets import get_student_record_summary
        record = get_student_record_summary("Zain")
        assert record is not None
        assert record["name"] == "Zain"
        assert record["grade"] == "8th Grade"
        assert record["academic_scores"]["Information Technology"] == 98
        assert "Robotics Team Captain" in record["extracurricular"]

    def test_get_presets_by_category(self):
        cats = get_presets_by_category()
        assert "Science" in cats
        assert "Mathematics" in cats
        assert "Computer Science" in cats
        assert "Study Skills" in cats
        assert "General" in cats
        assert "Student Reports" in cats

    def test_llm_wrapper_serves_preset(self):
        # Verify GroqClientWrapper yields preset without calling remote API
        cfg = LLMConfig(api_key="")  # empty key
        llm = GroqClientWrapper(cfg)

        responses = list(llm.stream_sentences("what is an algorithm"))
        assert len(responses) == 1
        assert "step-by-step" in responses[0].lower()
