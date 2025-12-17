"""BERTScore evaluation metrics for summary optimization.

Provides BERTScore evaluation replacing ROUGE with semantic-aware metrics
and composite scoring for Bayesian optimization.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from evaluate import load
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class BERTScoreEvaluator:
    """BERTScore evaluator for summary quality assessment."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-xlarge-mnli",
        batch_size: int = 16,
        rescale: bool = True,
        device: Optional[str] = None
    ):
        """Initialize BERTScore evaluator.

        Args:
            model_name: BERTScore model to use
            batch_size: Batch size for computation
            rescale: Whether to rescale scores to [0, 1]
            device: Device to run on (auto-detected if None)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.rescale = rescale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load BERTScore metric
        try:
            self.bertscore = load("bertscore")
        except Exception as e:
            logger.error(f"Failed to load BERTScore metric: {e}")
            raise

        # Load tokenizer for length analysis
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {model_name}: {e}")
            self.tokenizer = None

        logger.info(f"BERTScore evaluator initialized with model: {model_name}")

    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """Compute BERTScore metrics.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            verbose: Whether to show progress

        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) "
                f"and references ({len(references)}) must match"
            )

        if not predictions:
            return {"precision": [], "recall": [], "f1": []}

        # Compute BERTScore
        try:
            results = self.bertscore.compute(
                predictions=predictions,
                references=references,
                model_type=self.model_name,
                batch_size=self.batch_size,
                rescale_with_baseline=self.rescale,
                verbose=verbose
            )

            return {
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"]
            }

        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            # Return empty scores on error
            return {"precision": [], "recall": [], "f1": []}

    def compute_composite_score(
        self,
        precision: List[float],
        recall: List[float],
        f1: List[float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute composite BERTScore for optimization.

        Args:
            precision: BERTScore precision values
            recall: BERTScore recall values
            f1: BERTScore F1 values
            weights: Optional custom weights

        Returns:
            Composite score
        """
        if not precision or not recall or not f1:
            return 0.0

        # Default weights: emphasis on F1 (balance) and Precision (faithfulness)
        default_weights = {"precision": 0.3, "recall": 0.3, "f1": 0.4}
        weights = weights or default_weights

        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Compute averages
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)

        # Compute weighted composite
        composite = (
            weights["precision"] * avg_precision +
            weights["recall"] * avg_recall +
            weights["f1"] * avg_f1
        )

        return float(composite)

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            return_detailed: Whether to return detailed per-sample scores

        Returns:
            Dictionary with evaluation metrics
        """
        # Compute BERTScore
        scores = self.compute_bertscore(predictions, references)

        if not scores["precision"]:
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "bertscore_composite": 0.0,
                "num_samples": 0
            }

        # Compute composite score
        composite = self.compute_composite_score(
            scores["precision"], scores["recall"], scores["f1"]
        )

        result = {
            "bertscore_precision": float(np.mean(scores["precision"])),
            "bertscore_recall": float(np.mean(scores["recall"])),
            "bertscore_f1": float(np.mean(scores["f1"])),
            "bertscore_composite": float(composite),
            "num_samples": len(predictions)
        }

        if return_detailed:
            result.update({
                "precision_scores": scores["precision"],
                "recall_scores": scores["recall"],
                "f1_scores": scores["f1"]
            })

        # Add additional analysis if tokenizer available
        if self.tokenizer:
            result.update(self._analyze_generations(predictions, references))

        return result

    def _analyze_generations(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Analyze generation quality beyond BERTScore.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries

        Returns:
            Dictionary with analysis metrics
        """
        if not self.tokenizer:
            return {}

        analysis = {}

        try:
            # Length statistics
            pred_lengths = [len(self.tokenizer.encode(p)) for p in predictions]
            ref_lengths = [len(self.tokenizer.encode(r)) for r in references]

            analysis.update({
                "avg_pred_length": float(np.mean(pred_lengths)),
                "avg_ref_length": float(np.mean(ref_lengths)),
                "length_ratio": float(np.mean(pred_lengths) / np.mean(ref_lengths))
            })

            # Empty/short generation detection
            empty_preds = sum(1 for p in predictions if len(p.strip()) == 0)
            short_preds = sum(1 for p in predictions if len(p.split()) < 5)

            analysis.update({
                "empty_predictions": empty_preds,
                "short_predictions": short_preds,
                "empty_prediction_rate": float(empty_preds / len(predictions)),
                "short_prediction_rate": float(short_preds / len(predictions))
            })

            # Repetition analysis
            repetition_scores = []
            for pred in predictions:
                words = pred.split()
                if len(words) > 0:
                    unique_ratio = len(set(words)) / len(words)
                    repetition_scores.append(unique_ratio)

            if repetition_scores:
                analysis["avg_unique_token_ratio"] = float(np.mean(repetition_scores))

        except Exception as e:
            logger.warning(f"Error in generation analysis: {e}")

        return analysis

    def get_optimization_objective(
        self,
        predictions: List[str],
        references: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Get optimization objective value for Bayesian optimization.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            weights: Optional custom weights for composite score

        Returns:
            Objective value to maximize
        """
        scores = self.compute_bertscore(predictions, references)

        if not scores["f1"]:
            return 0.0

        # Compute composite score
        composite = self.compute_composite_score(
            scores["precision"], scores["recall"], scores["f1"], weights
        )

        # Penalize empty or very short generations
        penalty = 0.0
        for pred in predictions:
            if len(pred.strip()) == 0:
                penalty += 0.5  # Heavy penalty for empty
            elif len(pred.split()) < 5:
                penalty += 0.1  # Light penalty for very short

        penalty_rate = penalty / len(predictions)
        final_score = max(0.0, composite - penalty_rate)

        return float(final_score)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "eval"
    ) -> Dict[str, float]:
        """Format metrics for logging.

        Args:
            metrics: Metrics dictionary
            step: Optional step number
            prefix: Prefix for metric names

        Returns:
            Formatted metrics dictionary
        """
        formatted = {}
        for key, value in metrics.items():
            formatted_key = f"{prefix}/{key}" if prefix else key
            formatted[formatted_key] = value

        if step is not None:
            formatted["step"] = step

        return formatted


def create_evaluator(config) -> BERTScoreEvaluator:
    """Create BERTScore evaluator from configuration.

    Args:
        config: Configuration object with BERTScore parameters

    Returns:
        Configured BERTScore evaluator
    """
    return BERTScoreEvaluator(
        model_name=getattr(config, 'bertscore_model', "microsoft/deberta-xlarge-mnli"),
        batch_size=getattr(config, 'bertscore_batch_size', 16),
        rescale=getattr(config, 'bertscore_rescale', True)
    )