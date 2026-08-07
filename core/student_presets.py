"""
core/student_presets.py — Preset Q&A knowledge bank and intent matcher for students.

Provides:
  - Curated academic Q&As across Science, Math, Computer Science, Study Skills, and Creator info.
  - High-accuracy fuzzy normalization and intent matching engine.
  - Fast local lookup to eliminate LLM latency for frequent student questions.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)


class PresetItem(TypedDict):
    id: str
    category: str
    question: str
    answer: str
    triggers: list[str]


# ---------------------------------------------------------------------------
# Student Preset Knowledge Bank
# ---------------------------------------------------------------------------

STUDENT_PRESETS: list[PresetItem] = [
    # ── Science & Physics ────────────────────────────────────────────────
    {
        "id": "photosynthesis",
        "category": "Science",
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create food and release oxygen.",
        "triggers": [
            "photosynthesis", "define photosynthesis", "what is photosynthesis",
            "explain photosynthesis", "how plants make food", "process of photosynthesis"
        ],
    },
    {
        "id": "newton_first_law",
        "category": "Science",
        "question": "What is Newton's first law of motion?",
        "answer": "Newton's first law states that an object remains at rest or in uniform motion unless acted upon by an external force.",
        "triggers": [
            "newton first law", "newtons first law", "first law of motion",
            "law of inertia", "what is newton's first law", "explain newton first law"
        ],
    },
    {
        "id": "atom_structure",
        "category": "Science",
        "question": "What is the structure of an atom?",
        "answer": "An atom consists of a central nucleus containing protons and neutrons, surrounded by orbiting electrons.",
        "triggers": [
            "structure of atom", "structure of an atom", "what is an atom",
            "atom structure", "parts of an atom", "what does an atom consist of"
        ],
    },
    {
        "id": "water_formula",
        "category": "Science",
        "question": "What is the chemical formula for water?",
        "answer": "The chemical formula for water is H2O, which means it consists of two hydrogen atoms bonded to one oxygen atom.",
        "triggers": [
            "chemical formula for water", "formula of water", "water chemical formula",
            "what is water made of", "h2o formula", "formula for water"
        ],
    },
    {
        "id": "gravity",
        "category": "Science",
        "question": "What is gravity?",
        "answer": "Gravity is the fundamental force of attraction that pulls objects towards the center of Earth or any physical body with mass.",
        "triggers": [
            "what is gravity", "define gravity", "explain gravity",
            "force of gravity", "how gravity works"
        ],
    },
    {
        "id": "solar_system",
        "category": "Science",
        "question": "What are the planets in our solar system?",
        "answer": "Our solar system has eight planets in order from the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
        "triggers": [
            "planets in solar system", "eight planets", "solar system planets",
            "list planets", "names of planets", "what are the planets"
        ],
    },

    # ── Mathematics ──────────────────────────────────────────────────────
    {
        "id": "pythagorean_theorem",
        "category": "Mathematics",
        "question": "What is the Pythagorean theorem?",
        "answer": "The Pythagorean theorem states that in a right-angled triangle, a squared plus b squared equals c squared, where c is the hypotenuse.",
        "triggers": [
            "pythagorean theorem", "pythagoras theorem", "a squared plus b squared",
            "what is pythagorean theorem", "explain pythagoras theorem"
        ],
    },
    {
        "id": "area_of_circle",
        "category": "Mathematics",
        "question": "What is the formula for the area of a circle?",
        "answer": "The area of a circle is calculated using the formula pi times r squared, where r is the radius of the circle.",
        "triggers": [
            "area of a circle", "area of circle", "formula for area of circle",
            "how to find area of circle", "circle area formula"
        ],
    },
    {
        "id": "prime_number",
        "category": "Mathematics",
        "question": "What is a prime number?",
        "answer": "A prime number is a natural number greater than 1 that has exactly two factors: 1 and itself, such as 2, 3, 5, and 7.",
        "triggers": [
            "prime number", "what is a prime number", "define prime number",
            "examples of prime numbers", "prime numbers definition"
        ],
    },
    {
        "id": "value_of_pi",
        "category": "Mathematics",
        "question": "What is the value of Pi?",
        "answer": "Pi is a mathematical constant approximately equal to 3.14159, representing the ratio of a circle's circumference to its diameter.",
        "triggers": [
            "value of pi", "what is pi", "pi value",
            "value of pie", "number pi", "what is the value of pi"
        ],
    },
    {
        "id": "quadratic_formula",
        "category": "Mathematics",
        "question": "What is the quadratic formula?",
        "answer": "The quadratic formula solves a x squared plus b x plus c equals zero as x equals negative b plus or minus square root of b squared minus 4ac, all over 2a.",
        "triggers": [
            "quadratic formula", "formula for quadratic equation",
            "what is quadratic formula", "solve quadratic equation"
        ],
    },

    # ── Computer Science & AI ──────────────────────────────────────────────
    {
        "id": "python_lang",
        "category": "Computer Science",
        "question": "What is Python programming language?",
        "answer": "Python is a high-level, easy-to-read programming language widely used for artificial intelligence, web development, and data science.",
        "triggers": [
            "what is python", "python programming", "python language",
            "why use python", "define python"
        ],
    },
    {
        "id": "algorithm",
        "category": "Computer Science",
        "question": "What is an algorithm?",
        "answer": "An algorithm is a step-by-step set of instructions or rules followed by a computer or person to solve a specific problem.",
        "triggers": [
            "what is an algorithm", "define algorithm", "explain algorithm",
            "meaning of algorithm", "algorithm definition"
        ],
    },
    {
        "id": "artificial_intelligence",
        "category": "Computer Science",
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence, or AI, is the technology that enables computers and software to simulate human intelligence, learning, and reasoning.",
        "triggers": [
            "what is artificial intelligence", "what is ai", "define artificial intelligence",
            "explain ai", "meaning of ai"
        ],
    },
    {
        "id": "binary_system",
        "category": "Computer Science",
        "question": "What is the binary number system?",
        "answer": "Binary is a base-2 number system that represents all computer data and code using only two numbers: zero and one.",
        "triggers": [
            "what is binary", "binary system", "binary code",
            "how binary works", "binary numbers"
        ],
    },

    # ── Study Skills & Productivity ───────────────────────────────────────
    {
        "id": "study_effectively",
        "category": "Study Skills",
        "question": "How can I study effectively?",
        "answer": "To study effectively, practice active recall, space out your study sessions over time, summarize concepts in your own words, and take regular short breaks.",
        "triggers": [
            "how to study effectively", "study tips", "how can i study better",
            "best study methods", "effective studying", "how to prepare for exams"
        ],
    },
    {
        "id": "pomodoro_technique",
        "category": "Study Skills",
        "question": "What is the Pomodoro technique?",
        "answer": "The Pomodoro technique is a time management method where you focus on studying for 25 minutes, followed by a 5-minute break.",
        "triggers": [
            "pomodoro technique", "what is pomodoro", "pomodoro method",
            "pomodoro study", "explain pomodoro"
        ],
    },
    {
        "id": "note_taking",
        "category": "Study Skills",
        "question": "How do I take good study notes?",
        "answer": "Use structured methods like Cornell notes or mind maps, highlight key definitions, and write summaries in your own words right after class.",
        "triggers": [
            "how to take notes", "good note taking", "note taking tips",
            "taking study notes", "how to take good notes"
        ],
    },

    # ── Student Reports & Academic Performance ────────────────────────────
    {
        "id": "zain_performance",
        "category": "Student Reports",
        "question": "What is the previous month performance of my son Zain?",
        "answer": "Last month, Zain excelled with 95% in Science and 98% in Information Technology while captaining the Robotics team. To improve further, he should focus on timed English essay writing and exam time allocation.",
        "triggers": [
            "previous month performance of my son zain",
            "previous month performance of zain",
            "performance of my son zain",
            "how is zain performing",
            "how is my son zain doing",
            "zain performance report",
            "zain academic performance",
            "performance of zain",
            "zains report",
            "zain report"
        ],
    },
    {
        "id": "zain_strengths_weaknesses",
        "category": "Student Reports",
        "question": "What are Zain's strengths and weaknesses?",
        "answer": "Zain's strengths are analytical problem-solving, coding, and robotics leadership. His key area for growth is time management during written exams.",
        "triggers": [
            "zain strengths and weaknesses",
            "strengths and weaknesses of zain",
            "what are zains strengths",
            "what are zain's weaknesses",
            "zain strengths",
            "zain weaknesses"
        ],
    },

    # ── School & Assistant Capabilities ──────────────────────────────────
    {
        "id": "who_created_you",
        "category": "General",
        "question": "Who created you?",
        "answer": "I am Nyra, an AI voice assistant created by the talented students of Al Irshad Public School under the guidance and mentorship of Aitute.",
        "triggers": [
            "who created you", "who made you", "who built you",
            "who is your creator", "which school made you"
        ],
    },
    {
        "id": "what_can_you_do",
        "category": "General",
        "question": "What can you help me with?",
        "answer": "I can help you study, explain math and science concepts, share effective learning techniques, and answer your homework and quiz questions!",
        "triggers": [
            "what can you do", "what can you help me with", "how can you help me",
            "what are your features", "what can nyra do"
        ],
    },
]

# Structured Student Database for Extensible Record Queries
STUDENT_RECORDS = {
    "zain": {
        "name": "Zain",
        "grade": "8th Grade",
        "academic_scores": {
            "Information Technology": 98,
            "Science": 95,
            "Mathematics": 92,
            "English": 84,
        },
        "extracurricular": [
            "Robotics Team Captain",
            "School Football Team Striker",
            "Coding Club Lead",
        ],
        "strengths": [
            "Analytical problem solving",
            "High interest in coding & robotics",
            "Strong team leadership",
        ],
        "weaknesses": [
            "Time management during long written exams",
            "English essay formatting and structure",
        ],
        "summary": "Zain achieved an outstanding 95% in Science and 98% in CS last month, showing exceptional logic and robotics leadership.",
        "improvement_advice": "To reach his full potential, Zain should practice timed essay writing and pace himself carefully during written exams.",
    }
}


# ---------------------------------------------------------------------------
# Query Normalization & Intent Matching
# ---------------------------------------------------------------------------

_FILLER_WORDS_RE = re.compile(
    r"\b(can you|please|tell me|could you|i want to know|hey|nyra|nira|neera|naira|ora|aura|what is|whats|what's|explain|define|about|the)\b",
    re.IGNORECASE,
)


def normalize_query(text: str) -> str:
    """Clean and normalize a user query string for accurate intent matching."""
    if not text:
        return ""
    text = text.lower().strip()
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_preset_answer(user_query: str) -> Optional[str]:
    """
    Search student presets for a matching question or intent.

    Matching strategy:
      1. Direct normalized trigger match
      2. Key trigger phrase substring match
      3. Token set similarity match above threshold

    Returns:
      Answer string if a high-confidence match is found, else None.
    """
    if not user_query or not user_query.strip():
        return None

    norm_query = normalize_query(user_query)
    query_tokens = set(norm_query.split())

    # Stripped filler normalized query for core phrase checking
    core_query = _FILLER_WORDS_RE.sub("", norm_query)
    core_query = re.sub(r"\s+", " ", core_query).strip()

    best_match: Optional[PresetItem] = None
    best_score: float = 0.0

    for preset in STUDENT_PRESETS:
        for trigger in preset["triggers"]:
            norm_trigger = normalize_query(trigger)

            # Strategy 1: Exact match on normalized query or core query
            if norm_query == norm_trigger or core_query == norm_trigger:
                logger.info("Student preset EXACT match found for '%s': '%s'", user_query, preset["id"])
                return preset["answer"]

            # Strategy 2: Trigger as sub-phrase in user query
            if norm_trigger and (norm_trigger in norm_query or norm_trigger in core_query):
                logger.info("Student preset SUBSTRING match found for '%s': '%s'", user_query, preset["id"])
                return preset["answer"]

            # Strategy 3: Jaccard token set similarity
            trigger_tokens = set(norm_trigger.split())
            if not trigger_tokens:
                continue

            intersection = query_tokens.intersection(trigger_tokens)
            if intersection:
                score = len(intersection) / float(len(trigger_tokens))
                if score > best_score:
                    best_score = score
                    best_match = preset

    # High-confidence fallback for token similarity (>0.8 overlap)
    if best_match and best_score >= 0.8:
        logger.info("Student preset SIMILARITY match found (score=%.2f) for '%s': '%s'", best_score, user_query, best_match["id"])
        return best_match["answer"]

    return None


def get_presets_by_category() -> dict[str, list[PresetItem]]:
    """Group preset questions by category for UI cards/chips."""
    categories: dict[str, list[PresetItem]] = {}
    for item in STUDENT_PRESETS:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    return categories


def get_student_record_summary(name: str) -> Optional[dict]:
    """Retrieve full structured report record for a student by name."""
    if not name:
        return None
    key = name.strip().lower()
    return STUDENT_RECORDS.get(key)
