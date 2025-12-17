"""Enhanced wandb tracking for ELM summary optimization.

Provides hyperparameter sweep tracking, trial visualization setup,
hyperparameter importance analysis, and comprehensive logging.
"""

import wandb
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import optuna
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class WandbTracker:
    """Enhanced wandb tracker for optimization experiments."""

    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Initialize wandb tracker.

        Args:
            project_name: wandb project name
            experiment_name: Specific experiment name
            config: Configuration dictionary
            tags: List of tags for the run
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config or {}
        self.tags = tags or []
        self.run = None
        self.study_summary = {}

    def init_run(self):
        """Initialize wandb run."""
        try:
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                tags=self.tags,
                job_type="optimization"
            )
            logger.info(f"Initialized wandb run: {self.run.name}")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.run = None

    def log_trial_start(self, trial_number: int, params: Dict[str, Any]):
        """Log trial start.

        Args:
            trial_number: Trial number
            params: Trial hyperparameters
        """
        if not self.run:
            return

        # Log trial parameters
        self.run.log({
            f"trial_{trial_number}/params/{k}": v for k, v in params.items()
        }, commit=False)

        # Log trial status
        self.run.log({
            "trial_number": trial_number,
            "trial_status": "started",
            "active_trials": 1
        })

        logger.info(f"Logged trial {trial_number} start to wandb")

    def log_trial_complete(
        self,
        trial_number: int,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        training_time: float
    ):
        """Log trial completion.

        Args:
            trial_number: Trial number
            params: Trial hyperparameters
            metrics: Trial metrics
            training_time: Training time in minutes
        """
        if not self.run:
            return

        # Core metrics
        log_data = {
            f"trial_{trial_number}/metrics/objective": metrics.get("objective_value", 0.0),
            f"trial_{trial_number}/metrics/bertscore_composite": metrics.get("bertscore_composite", 0.0),
            f"trial_{trial_number}/metrics/bertscore_f1": metrics.get("bertscore_f1", 0.0),
            f"trial_{trial_number}/metrics/bertscore_precision": metrics.get("bertscore_precision", 0.0),
            f"trial_{trial_number}/metrics/bertscore_recall": metrics.get("bertscore_recall", 0.0),
            f"trial_{trial_number}/metrics/val_loss": metrics.get("val_loss", 0.0),
            f"trial_{trial_number}/time/training_minutes": training_time,
            "trial_status": "completed",
            "active_trials": 0,
            "completed_trials": trial_number + 1
        }

        # Log hyperparameters
        for param_name, param_value in params.items():
            log_data[f"trial_{trial_number}/params/{param_name}"] = param_value

        # Log interaction terms for analysis
        if "learning_rate" in params and "text_drift_weight" in params:
            log_data[f"trial_{trial_number}/interactions/lr_drift_product"] = (
                params["learning_rate"] * params["text_drift_weight"]
            )

        if "hidden_dim" in params and "batch_size" in params:
            log_data[f"trial_{trial_number}/interactions/hidden_batch_ratio"] = (
                params["hidden_dim"] / params["batch_size"]
            )

        self.run.log(log_data)

        # Store for summary
        self.study_summary[f"trial_{trial_number}"] = {
            "params": params,
            "metrics": metrics,
            "training_time": training_time
        }

        logger.info(f"Logged trial {trial_number} completion to wandb")

    def log_trial_pruned(self, trial_number: int, params: Dict[str, Any]):
        """Log trial pruning.

        Args:
            trial_number: Trial number
            params: Trial hyperparameters
        """
        if not self.run:
            return

        self.run.log({
            "trial_number": trial_number,
            "trial_status": "pruned",
            "active_trials": 0,
            "pruned_trials": 1
        })

        # Log pruned trial parameters
        for param_name, param_value in params.items():
            self.run.log({
                f"trial_{trial_number}/params/{param_name}": param_value
            }, commit=False)

        logger.info(f"Logged trial {trial_number} pruning to wandb")

    def log_hyperparameter_importance(self, importance: Dict[str, float]):
        """Log hyperparameter importance analysis.

        Args:
            importance: Dictionary of parameter importance scores
        """
        if not self.run:
            return

        # Log as table
        importance_table = wandb.Table(
            columns=["parameter", "importance"],
            data=[[k, v] for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)]
        )

        self.run.log({
            "hyperparameter_importance": importance_table,
            "optimization/total_parameters": len(importance)
        })

        # Create bar chart
        if importance:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(importance.keys()),
                    y=list(importance.values()),
                    text=list(importance.values()),
                    texttemplate='%{text:.3f}',
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Hyperparameter Importance",
                xaxis_title="Parameter",
                yaxis_title="Importance Score",
                showlegend=False
            )

            self.run.log({"hyperparameter_importance_chart": wandb.Plotly(fig)})

        logger.info("Logged hyperparameter importance to wandb")

    def log_optimization_summary(self, study: optuna.Study, optimization_time: float):
        """Log optimization summary.

        Args:
            study: Completed Optuna study
            optimization_time: Total optimization time in hours
        """
        if not self.run:
            return

        # Calculate summary statistics
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        # Objective statistics
        if completed_trials:
            objectives = [t.value for t in completed_trials]
            summary_stats = {
                "best_value": study.best_value,
                "mean_objective": sum(objectives) / len(objectives),
                "std_objective": (sum((x - sum(objectives)/len(objectives))**2 for x in objectives) / len(objectives))**0.5,
                "min_objective": min(objectives),
                "max_objective": max(objectives)
            }
        else:
            summary_stats = {}

        # Log summary
        summary_data = {
            "optimization/hours_spent": optimization_time,
            "optimization/total_trials": len(study.trials),
            "optimization/completed_trials": len(completed_trials),
            "optimization/pruned_trials": len(pruned_trials),
            "optimization/failed_trials": len(failed_trials),
            "optimization/pruning_rate": len(pruned_trials) / len(study.trials) if study.trials else 0,
        }

        if summary_stats:
            summary_data.update({
                f"summary/{k}": v for k, v in summary_stats.items()
            })

        self.run.log(summary_data)

        # Log optimization history
        if completed_trials:
            self._create_optimization_plots(completed_trials)

        logger.info("Logged optimization summary to wandb")

    def _create_optimization_plots(self, completed_trials: List[optuna.Trial]):
        """Create optimization visualization plots.

        Args:
            completed_trials: List of completed trials
        """
        if not self.run or not completed_trials:
            return

        # Extract data
        trial_numbers = [t.number for t in completed_trials]
        objectives = [t.value for t in completed_trials]

        # Optimization history
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=trial_numbers,
            y=objectives,
            mode='markers+lines',
            name='Objective Value',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        # Add best value line
        best_so_far = []
        current_best = float('-inf')
        for obj in objectives:
            if obj > current_best:
                current_best = obj
            best_so_far.append(current_best)

        fig1.add_trace(go.Scatter(
            x=trial_numbers,
            y=best_so_far,
            mode='lines',
            name='Best So Far',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig1.update_layout(
            title="Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Objective Value (BERTScore Composite)",
            hovermode='x unified'
        )

        self.run.log({"optimization_history": wandb.Plotly(fig1)})

        # Parameter relationships
        if len(completed_trials) > 5:  # Only if enough data
            self._create_parameter_plots(completed_trials)

    def _create_parameter_plots(self, completed_trials: List[optuna.Trial]):
        """Create parameter relationship plots.

        Args:
            completed_trials: List of completed trials
        """
        # Common parameters to visualize
        key_params = ['learning_rate', 'hidden_dim', 'batch_size', 'text_drift_weight', 'dropout_rate']

        for param in key_params:
            if param in completed_trials[0].params:
                values = [t.params[param] for t in completed_trials]
                objectives = [t.value for t in completed_trials]

                # Scatter plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=values,
                    y=objectives,
                    mode='markers',
                    text=[f"Trial {t.number}" for t in completed_trials],
                    marker=dict(
                        size=8,
                        color=objectives,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Objective Value")
                    )
                ))

                fig.update_layout(
                    title=f"Objective vs {param}",
                    xaxis_title=param,
                    yaxis_title="Objective Value"
                )

                self.run.log({f"parameter_objective_{param}": wandb.Plotly(fig)})

    def log_best_model_config(self, best_config: Dict[str, Any]):
        """Log best model configuration.

        Args:
            best_config: Best configuration found
        """
        if not self.run:
            return

        # Log as artifact
        config_path = Path("best_config.json")
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)

        artifact = wandb.Artifact("best_config", type="config")
        artifact.add_file(str(config_path))
        self.run.log_artifact(artifact)

        # Log config summary
        config_summary = {
            f"best_config/{k}": v for k, v in best_config.items()
            if isinstance(v, (int, float, str, bool))
        }

        self.run.log(config_summary)

        logger.info("Logged best configuration to wandb")

    def finish(self):
        """Finish wandb run."""
        if self.run:
            self.run.finish()
            logger.info("Finished wandb run")


def create_wandb_tracker(
    project_name: str,
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    enable: bool = True
) -> WandbTracker:
    """Create wandb tracker.

    Args:
        project_name: wandb project name
        experiment_name: Experiment name
        config: Configuration dictionary
        enable: Whether to enable wandb tracking

    Returns:
        WandbTracker instance
    """
    if not enable:
        logger.info("wandb tracking disabled")
        return WandbTracker(project_name, experiment_name, config)

    tracker = WandbTracker(
        project_name=project_name,
        experiment_name=experiment_name,
        config=config,
        tags=["elm-optimization", "summary-training", "bayesian-optimization"]
    )

    tracker.init_run()
    return tracker