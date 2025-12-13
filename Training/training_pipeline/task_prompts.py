"""Task prompt templates for ELM training.

These prompts MUST match exactly with Data_Synthesis/synthesis_pipeline/task_registry.py
to ensure consistency between data generation and training.

Phase I: Single-text tasks only (13 tasks)
Phase II: Pair-based tasks (compare, hypothetical) will be added
"""

# Single-text task prompts (13 tasks) - extracted from task_registry.py
SINGLE_TEXT_PROMPTS = {
    # FACTUAL TASKS (T=0.3, top_p=0.85)
    "keywords": """Extract 5-7 key concepts from this text as a list.
Focus on the main ideas, technical terms, and important entities.

Text:
{text}

Key Concepts:""",

    "category": """Identify the academic field or domain of this text.
Provide a brief justification for your classification.

Text:
{text}

Academic Field and Justification:""",

    "questions": """Generate 3 questions that this text could answer.
Questions should be specific and directly answerable from the content.

Text:
{text}

Questions:""",

    # DESCRIPTIVE TASKS (T=0.5, top_p=0.9)
    "summary": """Write a concise summary of this text in 2-3 sentences.
Capture the main topic and key points.

Text:
{text}

Summary:""",

    "describe": """Provide a detailed description of the content and themes in this text.
Cover the main ideas, structure, and any notable aspects.

Text:
{text}

Detailed Description:""",

    "explain_beginner": """Explain this text to someone with no background in this topic.
Use simple language and provide context for technical terms.

Text:
{text}

Beginner-Friendly Explanation:""",

    "explain_expert": """Explain this text with technical depth suitable for specialists in the field.
Use appropriate terminology and highlight nuances.

Text:
{text}

Expert-Level Explanation:""",

    "related_topics": """Suggest 5 related topics to this text with brief explanations of how they connect.

Text:
{text}

Related Topics:""",

    # CREATIVE TASKS (T=0.7, top_p=0.92)
    "characteristics_pos": """List 5 strengths or interesting aspects of the topic covered in this text.
Explain why each is noteworthy.

Text:
{text}

Strengths/Interesting Aspects:""",

    "characteristics_neg": """List 5 limitations, criticisms, or gaps related to the topic in this text.
Explain each point thoughtfully.

Text:
{text}

Limitations/Criticisms/Gaps:""",

    "style_academic": """Rewrite a description of this content in formal academic tone.
Use scholarly language and citation-ready phrasing.

Text:
{text}

Academic Description:""",

    "style_casual": """Rewrite a description of this content in a casual, conversational tone.
Make it engaging and accessible.

Text:
{text}

Casual Description:""",

    "counterfactual": """Imagine this topic were applied to {random_domain}.
What would change? How would it be different?

Text:
{text}

Counterfactual Analysis:""",
}


def get_single_text_prompt(task_type: str) -> str:
    """Get prompt template for a single-text task type.

    Args:
        task_type: Task type name (e.g., 'keywords', 'summary', etc.)

    Returns:
        Prompt template string

    Raises:
        ValueError: If task_type is unknown or is a pair-based task
    """
    if task_type not in SINGLE_TEXT_PROMPTS:
        if task_type in ["compare", "hypothetical"]:
            raise ValueError(
                f"Task '{task_type}' is pair-based and will be handled in Phase II. "
                "Use the hardcoded prompt from dataset.py for now."
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    return SINGLE_TEXT_PROMPTS[task_type]


def is_single_text_task(task_type: str) -> bool:
    """Check if task is a single-text task (Phase I).

    Args:
        task_type: Task type name

    Returns:
        True if task is single-text, False if pair-based
    """
    return task_type in SINGLE_TEXT_PROMPTS
