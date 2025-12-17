"""Bayesian Optimization for ELM summary training.

Main optimizer class using Optuna with 2-epoch trial management,
hyperparameter sampling logic including drift parameters,
and result aggregation based on BERTScore.
"""

import optuna
import torch
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, asdict

from .bayesian_config import (
    suggest_hyperparameters,
    validate_hyperparameters,
    create_study,
    format_trial_params
)
from .bertscore_metrics import BERTScoreEvaluator, create_evaluator
from .wandb_tracker import create_wandb_tracker

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Container for trial results."""
    trial_number: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    objective_value: float
    training_time: float
    status: str
    pruned: bool = False
    error_message: Optional[str] = None


class BayesianOptimizer:
    """Bayesian optimizer for ELM summary training."""

    def __init__(
        self,
        study_name: str,
        storage_url: Optional[str] = None,
        max_trials: int = 50,
        timeout_hours: float = 24.0,
        n_jobs: int = 1,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None
    ):
        """Initialize Bayesian optimizer.

        Args:
            study_name: Name of the optimization study
            storage_url: Optional database URL for persistence
            max_trials: Maximum number of trials
            timeout_hours: Maximum time in hours
            n_jobs: Number of parallel jobs
            device: Device for training
            checkpoint_dir: Directory for trial checkpoints
        """
        self.study_name = study_name
        self.storage_url = storage_url
        self.max_trials = max_trials
        self.timeout_hours = timeout_hours
        self.n_jobs = n_jobs
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Initialize study
        self.study = create_study(
            study_name=study_name,
            direction="maximize",  # Maximize BERTScore composite
            storage_url=storage_url,
            load_if_exists=True
        )

        # Results storage
        self.results: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None

        # BERTScore evaluator
        self.evaluator = None

        # wandb tracker
        self.wandb_tracker = None

        logger.info(f"Bayesian optimizer initialized: {study_name}")

    def optimize(
        self,
        objective_func: callable,
        resume: bool = True
    ) -> optuna.Study:
        """Run Bayesian optimization.

        Args:
            objective_func: Objective function that takes trial and returns objective value
            resume: Whether to resume from existing study

        Returns:
            Completed Optuna study
        """
        logger.info(f"Starting optimization: max_trials={self.max_trials}, "
                   f"timeout={self.timeout_hours}h")

        start_time = time.time()
        timeout_seconds = self.timeout_hours * 3600

        try:
            # Run optimization
            self.study.optimize(
                objective_func,
                n_trials=self.max_trials,
                timeout=timeout_seconds,
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )

        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

        finally:
            # Log optimization summary
            elapsed_time = (time.time() - start_time) / 3600  # hours
            self._log_optimization_summary(elapsed_time)

            # Final wandb logging
            if self.wandb_tracker:
                self.wandb_tracker.log_hyperparameter_importance(
                    self.analyze_hyperparameter_importance()
                )
                self.wandb_tracker.log_optimization_summary(self.study, elapsed_time)
                if self.best_trial:
                    best_config = self.get_best_params()
                    self.wandb_tracker.log_best_model_config(best_config)

        return self.study

    def create_objective_function(
        self,
        train_func: callable,
        eval_func: callable,
        config_template: dict
    ) -> callable:
        """Create objective function for optimization.

        Args:
            train_func: Training function that takes config and returns metrics
            eval_func: Evaluation function that takes model and returns BERTScore
            config_template: Base configuration template

        Returns:
            Objective function compatible with Optuna
        """
        def objective(trial: optuna.Trial) -> float:
            try:
                # Suggest hyperparameters
                params = suggest_hyperparameters(trial)
                validated_params = validate_hyperparameters(params)

                # Create config for this trial
                config = self._create_trial_config(validated_params, config_template)

                # Log trial start
                logger.info(f"Trial {trial.number}: {format_trial_params(validated_params)}")

                # Report intermediate values for pruning (2-epoch constraint)
                trial.report(0.0, step=0)  # Start value

                # Train model with 2 epochs
                start_time = time.time()
                training_metrics = train_func(config, trial)
                training_time = time.time() - start_time

                # Report training progress for pruning check
                if training_metrics.get("val_bertscore", 0.0) < 0.3:
                    # Early pruning if very poor performance
                    trial.report(training_metrics.get("val_bertscore", 0.0), step=1)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                # Final evaluation
                final_metrics = eval_func(config)
                objective_value = final_metrics.get("bertscore_composite", 0.0)

                # Store trial result
                result = TrialResult(
                    trial_number=trial.number,
                    params=validated_params,
                    metrics=final_metrics,
                    objective_value=objective_value,
                    training_time=training_time,
                    status="completed"
                )
                self.results.append(result)

                # Update best trial
                if (self.best_trial is None or
                    objective_value > self.best_trial.objective_value):
                    self.best_trial = result

                # Log trial completion
                logger.info(
                    f"Trial {trial.number} completed: "
                    f"objective={objective_value:.4f}, "
                    f"time={training_time/60:.1f}min"
                )

                # Report final value for pruning
                trial.report(objective_value, step=2)

                return objective_value

            except optuna.exceptions.TrialPruned:
                # Handle pruning
                result = TrialResult(
                    trial_number=trial.number,
                    params=validated_params if 'validated_params' in locals() else {},
                    metrics={},
                    objective_value=0.0,
                    training_time=0.0,
                    status="pruned",
                    pruned=True
                )
                self.results.append(result)

                logger.info(f"Trial {trial.number} pruned")
                raise

            except Exception as e:
                # Handle errors
                error_msg = str(e)
                logger.error(f"Trial {trial.number} failed: {error_msg}")

                result = TrialResult(
                    trial_number=trial.number,
                    params=validated_params if 'validated_params' in locals() else {},
                    metrics={},
                    objective_value=0.0,
                    training_time=0.0,
                    status="failed",
                    error_message=error_msg
                )
                self.results.append(result)

                # Return very low score for failed trials
                return 0.0

        return objective

    def _create_trial_config(
        self,
        params: Dict[str, Any],
        template: dict
    ) -> Dict[str, Any]:
        """Create configuration for a specific trial.

        Args:
            params: Trial hyperparameters
            template: Base configuration template

        Returns:
            Complete configuration for the trial
        """
        config = template.copy()
        config.update(params)

        # Add trial-specific settings
        config.update({
            "summary_only": True,
            "max_epochs": 2,  # Fixed 2-epoch constraint
            "use_text_drift_loss": True,
            "device": self.device,
            "trial_id": len(self.results)
        })

        # Create trial-specific checkpoint directory
        if self.checkpoint_dir:
            trial_dir = self.checkpoint_dir / f"trial_{len(self.results)}"
            config["checkpoint_dir"] = trial_dir
            trial_dir.mkdir(parents=True, exist_ok=True)

        return config

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from optimization.

        Returns:
            Best hyperparameters found
        """
        if not self.study.best_trial:
            return {}

        return self.study.best_trial.params

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history for analysis.

        Returns:
            List of trial data for plotting/analysis
        """
        history = []

        for trial in self.study.trials:
            history.append({
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            })

        return history

    def analyze_hyperparameter_importance(self) -> Dict[str, float]:
        """Analyze hyperparameter importance.

        Returns:
            Dictionary of parameter importance scores
        """
        if len(self.study.trials) < 2:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.warning(f"Failed to compute hyperparameter importance: {e}")
            return {}

    def save_results(self, filepath: Path) -> None:
        """Save optimization results to file.

        Args:
            filepath: Path to save results
        """
        results_data = {
            "study_name": self.study_name,
            "best_params": self.get_best_params(),
            "best_value": self.study.best_value if self.study.best_trial else None,
            "n_trials": len(self.study.trials),
            "optimization_history": self.get_optimization_history(),
            "hyperparameter_importance": self.analyze_hyperparameter_importance(),
            "detailed_results": [asdict(result) for result in self.results]
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Optimization results saved to {filepath}")

    def load_results(self, filepath: Path) -> None:
        """Load optimization results from file.

        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            results_data = json.load(f)

        # Restore results (basic reconstruction)
        self.results = []
        for result_data in results_data.get("detailed_results", []):
            result = TrialResult(**result_data)
            self.results.append(result)

        # Find best trial
        if self.results:
            self.best_trial = max(
                [r for r in self.results if r.status == "completed"],
                key=lambda x: x.objective_value
            )

        logger.info(f"Optimization results loaded from {filepath}")

    def _log_optimization_summary(self, elapsed_time: float) -> None:
        """Log optimization summary.

        Args:
            elapsed_time: Time spent optimizing in hours
        """
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*60)

        logger.info(f"Study: {self.study_name}")
        logger.info(f"Total trials: {len(self.study.trials)}")
        logger.info(f"Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        logger.info(f"Pruned trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        logger.info(f"Failed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} hours")

        if self.study.best_trial:
            logger.info(f"\nBest trial: {self.study.best_trial.number}")
            logger.info(f"Best value: {self.study.best_value:.4f}")
            logger.info(f"Best params: {format_trial_params(self.study.best_trial.params)}")

        # Log hyperparameter importance
        importance = self.analyze_hyperparameter_importance()
        if importance:
            logger.info("\nHyperparameter Importance:")
            for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {param}: {score:.4f}")

        logger.info("="*60)


def create_optimizer(
    study_name: str,
    storage_url: Optional[str] = None,
    max_trials: int = 50,
    timeout_hours: float = 24.0,
    checkpoint_dir: Optional[Path] = None
) -> BayesianOptimizer:
    """Create Bayesian optimizer instance.

    Args:
        study_name: Name of the study
        storage_url: Optional database URL
        max_trials: Maximum number of trials
        timeout_hours: Maximum optimization time
        checkpoint_dir: Directory for checkpoints

    Returns:
        Configured Bayesian optimizer
    """
    return BayesianOptimizer(
        study_name=study_name,
        storage_url=storage_url,
        max_trials=max_trials,
        timeout_hours=timeout_hours,
        checkpoint_dir=checkpoint_dir
    )