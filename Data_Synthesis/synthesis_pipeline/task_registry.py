"""
Task type definitions and prompt templates for data synthesis.
"""

from typing import Dict, List
from .config import TaskCategory, TaskConfig


# Category A - Factual Tasks (T=0.3, top_p=0.85)
FACTUAL_TASKS = {
    "keywords": TaskConfig(
        name="keywords",
        category=TaskCategory.FACTUAL,
        temperature=0.3,
        top_p=0.85,
        min_tokens=25,
        prompt_template="""Extract 5-7 key concepts from this text as a list.
Focus on the main ideas, technical terms, and important entities.

Text:
{text}

Key Concepts:""",
        variations=2,
    ),
    "category": TaskConfig(
        name="category",
        category=TaskCategory.FACTUAL,
        temperature=0.3,
        top_p=0.85,
        min_tokens=40,
        prompt_template="""Identify the academic field or domain of this text.
Provide a brief justification for your classification.

Text:
{text}

Academic Field and Justification:""",
        variations=2,
    ),
    "questions": TaskConfig(
        name="questions",
        category=TaskCategory.FACTUAL,
        temperature=0.3,
        top_p=0.85,
        min_tokens=60,
        prompt_template="""Generate 3 questions that this text could answer.
Questions should be specific and directly answerable from the content.

Text:
{text}

Questions:""",
        variations=3,
    ),
}

# Category B - Descriptive Tasks (T=0.5, top_p=0.9)
DESCRIPTIVE_TASKS = {
    "summary": TaskConfig(
        name="summary",
        category=TaskCategory.DESCRIPTIVE,
        temperature=0.5,
        top_p=0.9,
        min_tokens=50,
        prompt_template="""Write a concise summary of this text in 2-3 sentences.
Capture the main topic and key points.

Text:
{text}

Summary:""",
        variations=2,
    ),
    "describe": TaskConfig(
        name="describe",
        category=TaskCategory.DESCRIPTIVE,
        temperature=0.5,
        top_p=0.9,
        min_tokens=120,
        prompt_template="""Provide a detailed description of the content and themes in this text.
Cover the main ideas, structure, and any notable aspects.

Text:
{text}

Detailed Description:""",
        variations=2,
    ),
    "explain_beginner": TaskConfig(
        name="explain_beginner",
        category=TaskCategory.DESCRIPTIVE,
        temperature=0.5,
        top_p=0.9,
        min_tokens=100,
        prompt_template="""Explain this text to someone with no background in this topic.
Use simple language and provide context for technical terms.

Text:
{text}

Beginner-Friendly Explanation:""",
        variations=2,
    ),
    "explain_expert": TaskConfig(
        name="explain_expert",
        category=TaskCategory.DESCRIPTIVE,
        temperature=0.5,
        top_p=0.9,
        min_tokens=100,
        prompt_template="""Explain this text with technical depth suitable for specialists in the field.
Use appropriate terminology and highlight nuances.

Text:
{text}

Expert-Level Explanation:""",
        variations=2,
    ),
    "related_topics": TaskConfig(
        name="related_topics",
        category=TaskCategory.DESCRIPTIVE,
        temperature=0.5,
        top_p=0.9,
        min_tokens=80,
        prompt_template="""Suggest 5 related topics to this text with brief explanations of how they connect.

Text:
{text}

Related Topics:""",
        variations=2,
    ),
}

# Category C - Creative/Hypothetical Tasks (T=0.7, top_p=0.92)
CREATIVE_TASKS = {
    "characteristics_pos": TaskConfig(
        name="characteristics_pos",
        category=TaskCategory.CREATIVE,
        temperature=0.7,
        top_p=0.92,
        min_tokens=80,
        prompt_template="""List 5 strengths or interesting aspects of the topic covered in this text.
Explain why each is noteworthy.

Text:
{text}

Strengths/Interesting Aspects:""",
        variations=2,
    ),
    "characteristics_neg": TaskConfig(
        name="characteristics_neg",
        category=TaskCategory.CREATIVE,
        temperature=0.7,
        top_p=0.92,
        min_tokens=80,
        prompt_template="""List 5 limitations, criticisms, or gaps related to the topic in this text.
Explain each point thoughtfully.

Text:
{text}

Limitations/Criticisms/Gaps:""",
        variations=2,
    ),
    "style_academic": TaskConfig(
        name="style_academic",
        category=TaskCategory.CREATIVE,
        temperature=0.7,
        top_p=0.92,
        min_tokens=100,
        prompt_template="""Rewrite a description of this content in formal academic tone.
Use scholarly language and citation-ready phrasing.

Text:
{text}

Academic Description:""",
        variations=2,
    ),
    "style_casual": TaskConfig(
        name="style_casual",
        category=TaskCategory.CREATIVE,
        temperature=0.7,
        top_p=0.92,
        min_tokens=100,
        prompt_template="""Rewrite a description of this content in a casual, conversational tone.
Make it engaging and accessible.

Text:
{text}

Casual Description:""",
        variations=2,
    ),
    "counterfactual": TaskConfig(
        name="counterfactual",
        category=TaskCategory.CREATIVE,
        temperature=0.7,
        top_p=0.92,
        min_tokens=100,
        prompt_template="""Imagine this topic were applied to {random_domain}.
What would change? How would it be different?

Text:
{text}

Counterfactual Analysis:""",
        variations=2,
    ),
}

# Category D - Pair-Based Tasks (require k-NN pairing)
PAIR_BASED_TASKS = {
    "compare": TaskConfig(
        name="compare",
        category=TaskCategory.PAIR_BASED,
        temperature=0.5,
        top_p=0.9,
        min_tokens=150,
        prompt_template="""Compare and contrast these two related texts.
Identify similarities, differences, and how they complement each other.

Text 1:
{text1}

Text 2:
{text2}

Comparison:""",
        variations=2,
        requires_pair=True,
        knn_k=5,
    ),
    "hypothetical": TaskConfig(
        name="hypothetical",
        category=TaskCategory.PAIR_BASED,
        temperature=0.7,
        top_p=0.92,
        min_tokens=120,
        prompt_template="""Describe the conceptual midpoint between these two texts.
What would a topic that combines elements of both look like?

Text 1 (weight: {alpha1:.2f}):
{text1}

Text 2 (weight: {alpha2:.2f}):
{text2}

Conceptual Midpoint:""",
        variations=2,
        requires_pair=True,
        knn_k=3,
        alpha_range=(0.3, 0.7),
    ),
}

# Random domains for counterfactual task
COUNTERFACTUAL_DOMAINS = [
    "healthcare", "education", "finance", "entertainment",
    "agriculture", "transportation", "construction", "retail",
    "manufacturing", "telecommunications", "hospitality", "energy",
    "aerospace", "marine biology", "archaeology", "culinary arts",
    "fashion design", "urban planning", "wildlife conservation",
    "renewable energy", "artificial intelligence", "quantum computing",
]


class TaskRegistry:
    """Registry for all task types."""

    def __init__(self):
        self.tasks: Dict[str, TaskConfig] = {}
        self._register_all_tasks()

    def _register_all_tasks(self):
        """Register all task types."""
        self.tasks.update(FACTUAL_TASKS)
        self.tasks.update(DESCRIPTIVE_TASKS)
        self.tasks.update(CREATIVE_TASKS)
        self.tasks.update(PAIR_BASED_TASKS)

    def get_task(self, name: str) -> TaskConfig:
        """Get task configuration by name.

        Args:
            name: Task name

        Returns:
            TaskConfig for the specified task

        Raises:
            ValueError: If task name is not found
        """
        if name not in self.tasks:
            raise ValueError(f"Unknown task: {name}")
        return self.tasks[name]

    def get_all_tasks(self) -> List[TaskConfig]:
        """Get all task configurations.

        Returns:
            List of all TaskConfig objects
        """
        return list(self.tasks.values())

    def get_tasks_by_category(self, category: TaskCategory) -> List[TaskConfig]:
        """Get tasks filtered by category.

        Args:
            category: TaskCategory to filter by

        Returns:
            List of TaskConfig objects in the category
        """
        return [t for t in self.tasks.values() if t.category == category]

    def get_single_text_tasks(self) -> List[TaskConfig]:
        """Get tasks that only require a single text input.

        Returns:
            List of single-text TaskConfig objects
        """
        return [t for t in self.tasks.values() if not t.requires_pair]

    def get_pair_tasks(self) -> List[TaskConfig]:
        """Get tasks that require text pairs.

        Returns:
            List of pair-based TaskConfig objects
        """
        return [t for t in self.tasks.values() if t.requires_pair]
