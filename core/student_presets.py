"""
core/student_presets.py — Preset Q&A knowledge bank and intent matcher for students.

Provides:
  - Curated academic Q&As across Science, Math, Computer Science, Study Skills, and Creator info.
  - 3 to 4 answer variations per preset to avoid repetitive responses.
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
    answers: list[str]
    triggers: list[str]


# State tracker for rotating answer variations per preset ID
_PRESET_USAGE_TRACKER: dict[str, int] = {}


def reset_preset_tracker() -> None:
    """Reset the usage tracker (useful for unit testing)."""
    _PRESET_USAGE_TRACKER.clear()


# ---------------------------------------------------------------------------
# Student Preset Knowledge Bank (with 3-4 variations per question)
# ---------------------------------------------------------------------------

STUDENT_PRESETS: list[PresetItem] = [
    # ── Science & Physics ────────────────────────────────────────────────
    {
        "id": "photosynthesis",
        "category": "Science",
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create food and release oxygen.",
        "answers": [
            "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create food and release oxygen.",
            "In photosynthesis, plants convert light energy from the sun into chemical energy, turning carbon dioxide and water into glucose and oxygen.",
            "Plants make their own food through photosynthesis using sunlight, water from soil, and carbon dioxide from the air while generating fresh oxygen.",
            "Photosynthesis is how green plants capture sunlight with chlorophyll to produce sugars and oxygen, feeding themselves and sustaining life on Earth.",
        ],
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
        "answers": [
            "Newton's first law states that an object remains at rest or in uniform motion unless acted upon by an external force.",
            "Also known as the law of inertia, Newton's first law explains that an object won't change its speed or direction unless a net force compels it.",
            "According to Newton's first law of motion, things keep doing what they are doing—staying still or moving at constant speed—until a force interferes.",
            "Newton's first law describes inertia: a stationary object stays still, and a moving object keeps moving, unless pushed or pulled by an outside force.",
        ],
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
        "answers": [
            "An atom consists of a central nucleus containing protons and neutrons, surrounded by orbiting electrons.",
            "Atoms are made of three main subatomic particles: positively charged protons and neutral neutrons in the nucleus, with negatively charged electrons orbiting outside.",
            "The structure of an atom features a dense central core of protons and neutrons surrounded by a cloud of fast-moving electrons.",
            "Inside an atom, you'll find a central nucleus packed with protons and neutrons, encircled by tiny orbiting electrons.",
        ],
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
        "answers": [
            "The chemical formula for water is H2O, which means it consists of two hydrogen atoms bonded to one oxygen atom.",
            "Water is written chemically as H2O, indicating a molecule formed by two hydrogen atoms attached to a single oxygen atom.",
            "H2O is the chemical symbol for water, representing two parts hydrogen to one part oxygen in every water molecule.",
            "Every molecule of water has two hydrogen atoms and one oxygen atom, giving it the well-known formula H2O.",
        ],
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
        "answers": [
            "Gravity is the fundamental force of attraction that pulls objects towards the center of Earth or any physical body with mass.",
            "Gravity is an invisible natural force that pulls mass toward mass, keeping our feet on the ground and planets in orbit.",
            "Sir Isaac Newton described gravity as the attractive force between any two objects, which gets stronger with more mass and closer distance.",
            "Gravity is the force that gives weight to objects and pulls everything toward the center of mass, like Earth pulling objects down.",
        ],
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
        "answers": [
            "Our solar system has eight planets in order from the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            "Starting closest to the Sun, the eight planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            "There are eight planets orbiting the Sun: the four inner rocky planets—Mercury, Venus, Earth, Mars—and the four outer gas and ice giants—Jupiter, Saturn, Uranus, and Neptune.",
            "In order of distance from the Sun, our solar system's eight planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
        ],
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
        "answers": [
            "The Pythagorean theorem states that in a right-angled triangle, a squared plus b squared equals c squared, where c is the hypotenuse.",
            "In geometry, the Pythagorean theorem formula is a² + b² = c², relating the two legs of a right triangle to its longest side, the hypotenuse.",
            "The Pythagorean theorem tells us that squaring the two shorter sides of a right triangle and adding them gives the square of the hypotenuse.",
            "For any right-angled triangle, Pythagoras discovered that a squared plus b squared equals c squared.",
        ],
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
        "answers": [
            "The area of a circle is calculated using the formula pi times r squared, where r is the radius of the circle.",
            "To find the area of a circle, multiply Pi by the square of its radius, expressed as A = π r².",
            "The formula for circle area is Pi times radius squared. Measure the radius from center to edge, square it, and multiply by 3.14.",
            "Area equals pi r squared. You square the circle's radius and multiply by Pi to find the total internal surface area.",
        ],
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
        "answers": [
            "A prime number is a natural number greater than 1 that has exactly two factors: 1 and itself, such as 2, 3, 5, and 7.",
            "Prime numbers are whole numbers larger than 1 that cannot be divided evenly by any number except 1 and themselves.",
            "In math, a prime number has only two positive divisors: 1 and the number itself. Examples include 2, 3, 5, 7, 11, and 13.",
            "A prime number is any integer above 1 that is divisible only by 1 and itself, with 2 being the only even prime number.",
        ],
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
        "answers": [
            "Pi is a mathematical constant approximately equal to 3.14159, representing the ratio of a circle's circumference to its diameter.",
            "The value of Pi is roughly 3.14159 or 22 over 7. It measures the ratio of any circle's perimeter to its diameter.",
            "Pi is an irrational number commonly rounded to 3.14 or 3.14159, used extensively in geometry and physics calculations.",
            "Pi is approximately 3.14159, a mathematical constant that never ends or repeats.",
        ],
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
        "answers": [
            "The quadratic formula solves a x squared plus b x plus c equals zero as x equals negative b plus or minus square root of b squared minus 4ac, all over 2a.",
            "For any quadratic equation ax² + bx + c = 0, the formula is x equals -b ± √(b² - 4ac) divided by 2a.",
            "The quadratic formula calculates the roots of a second-degree polynomial equation as negative b plus or minus the square root of b squared minus 4ac, divided by 2a.",
            "To solve quadratic equations, use x = (-b ± √(b² - 4ac)) / (2a) to find the values where the equation equals zero.",
        ],
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
        "answers": [
            "Python is a high-level, easy-to-read programming language widely used for artificial intelligence, web development, and data science.",
            "Python is a versatile programming language known for clean syntax, clear code readability, and vast libraries for AI and software development.",
            "Python is one of the world's most popular coding languages, favored by beginners and experts alike for machine learning, automation, and web apps.",
            "Python is a powerful open-source programming language created by Guido van Rossum, prized for readable code and extensive ecosystem support.",
        ],
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
        "answers": [
            "An algorithm is a step-by-step set of instructions or rules followed by a computer or person to solve a specific problem.",
            "Think of an algorithm as a detailed recipe or procedure that gives explicit rules to achieve a target outcome or calculation.",
            "In computer science, an algorithm is a clear sequence of logical steps designed to process data and solve a task.",
            "An algorithm is an ordered set of problem-solving instructions that computers execute to process information efficiently.",
        ],
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
        "answers": [
            "Artificial Intelligence, or AI, is the technology that enables computers and software to simulate human intelligence, learning, and reasoning.",
            "AI refers to computer systems engineered to perform tasks that typically require human intellect, like speech recognition and decision making.",
            "Artificial Intelligence combines computer science and datasets to enable machine learning, problem-solving, and automated reasoning.",
            "AI is a branch of computer science focused on building smart machines capable of learning from data and carrying out human-like reasoning.",
        ],
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
        "answers": [
            "Binary is a base-2 number system that represents all computer data and code using only two numbers: zero and one.",
            "Computers talk in binary, a base-two counting system where all information is stored as sequences of 0s and 1s.",
            "The binary system uses 0 and 1 as bits to represent electrical off and on states inside computer processors.",
            "Binary is the fundamental language of computers, converting numbers, letters, and multimedia into combinations of zeroes and ones.",
        ],
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
        "answers": [
            "To study effectively, practice active recall, space out your study sessions over time, summarize concepts in your own words, and take regular short breaks.",
            "Boost your study efficiency with spaced repetition, teaching concepts to someone else, creating practice quizzes, and working in timed focus sessions.",
            "Effective studying relies on active learning: test yourself frequently, break complex topics into smaller chunks, and get plenty of rest before exams.",
            "Key study strategies include active recall testing, spacing out study days rather than cramming, and summarizing key ideas without looking at notes.",
        ],
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
        "answers": [
            "The Pomodoro technique is a time management method where you focus on studying for 25 minutes, followed by a 5-minute break.",
            "With Pomodoro, you work with full concentration for 25 minutes, take a 5-minute breather, and after 4 rounds take a longer 15 to 30 minute break.",
            "The Pomodoro method splits work into 25-minute intervals separated by short breaks, helping prevent mental fatigue and maintain deep focus.",
            "Invented by Francesco Cirillo, the Pomodoro technique uses 25-minute study blocks followed by quick 5-minute breaks to maximize productivity.",
        ],
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
        "answers": [
            "Use structured methods like Cornell notes or mind maps, highlight key definitions, and write summaries in your own words right after class.",
            "Great note-taking involves organizing ideas into headings, using bullet points and abbreviations, and reviewing your notes within 24 hours.",
            "To take effective notes, focus on main concepts rather than writing word-for-word, draw visual diagrams, and summarize key takeaways at the bottom.",
            "Try the Cornell note-taking system: divide your page into cue words, main notes, and a summary section for easy revision.",
        ],
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
        "answers": [
            "Last month, Zain excelled with 95% in Science and 98% in Information Technology while captaining the Robotics team. To improve further, he should focus on timed English essay writing and exam time allocation.",
            "Zain had an outstanding month! He scored 98% in IT and 95% in Science, leading the Coding Club and Robotics team. His main focus area now is practicing timed essay writing in English.",
            "According to Zain's monthly performance report, he achieved top marks in Computer Science (98%) and Science (95%). With practice on exam time management for English, he will be in top form.",
            "Zain demonstrated superb academic strength in Science and IT last month. He continues to shine in extracurricular robotics, with extra practice recommended for English essay formatting under timed conditions.",
        ],
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
        "answers": [
            "Zain's strengths are analytical problem-solving, coding, and robotics leadership. His key area for growth is time management during written exams.",
            "Zain excels in logical problem-solving, programming, and team leadership. He can improve by practicing time allocation on long written English exams.",
            "Zain's major strengths include high technical aptitude in IT and Science alongside great leadership skills. His primary growth goal is building speed in essay writing.",
            "Analytical thinking and coding are Zain's biggest strengths. To balance his profile, he's working on managing his time better during long written test papers.",
        ],
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
        "answer": "I am Nyra, an AI voice assistant created by the talented students of Al Irshad Central School under the guidance and mentorship of Aitute.",
        "answers": [
            "I am Nyra, an AI voice assistant created by the talented students of Al Irshad Central School under the guidance and mentorship of Aitute.",
            "I was built by the student innovation team at Al Irshad Central School, mentored by Aitute to bring smart AI assistance to learning.",
            "Nyra was developed as an educational project by students at Al Irshad Central School in collaboration with Aitute.",
            "The bright students of Al Irshad Central School designed and developed me with guidance from Aitute to serve as a personalized study companion.",
        ],
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
        "answers": [
            "I can help you study, explain math and science concepts, share effective learning techniques, and answer your homework and quiz questions!",
            "As your AI study assistant, I can explain academic concepts, quiz you on topics, offer study tips, and check student performance reports!",
            "I'm equipped to assist with science laws, mathematical formulas, computer science topics, study methods, and academic progress updates.",
            "I can assist with homework questions, break down complex STEM subjects, guide your study habits, and share student academic summaries!",
        ],
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


def get_preset_variation(preset: PresetItem) -> str:
    """Return a rotated variation of answer for the given preset item to prevent repetition."""
    answers = preset.get("answers")
    if answers and isinstance(answers, list) and len(answers) > 0:
        preset_id = preset["id"]
        last_idx = _PRESET_USAGE_TRACKER.get(preset_id, -1)
        next_idx = (last_idx + 1) % len(answers)
        _PRESET_USAGE_TRACKER[preset_id] = next_idx
        return answers[next_idx]
    return preset.get("answer", "")


def find_preset_answer(user_query: str) -> Optional[str]:
    """
    Search student presets for a matching question or intent.

    Matching strategy:
      1. Direct normalized trigger match
      2. Key trigger phrase substring match
      3. Token set similarity match above threshold

    Returns:
      Rotated answer variation string if a high-confidence match is found, else None.
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
                return get_preset_variation(preset)

            # Strategy 2: Trigger as sub-phrase in user query
            if norm_trigger and (norm_trigger in norm_query or norm_trigger in core_query):
                logger.info("Student preset SUBSTRING match found for '%s': '%s'", user_query, preset["id"])
                return get_preset_variation(preset)

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
        return get_preset_variation(best_match)

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
