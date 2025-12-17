"""Bayesian Optimization configuration for ELM summary training.

Defines the hyperparameter search space for optimization including drift loss parameters.
"""

from typing import Dict, Any
import optuna


def get_search_space() -> Dict[str, Any]:
    """Get the hyperparameter search space for Bayesian optimization.

    Returns:
        Dictionary containing parameter search space definitions
    """
    return {
        # Learning rate search space (log scale)
        "learning_rate": {
            "type": "log_uniform",
            "low": 1e-5,
            "high": 5e-3,
            "description": "Learning rate for AdamW optimizer"
        },

        # Hidden dimension (discrete choices)
        "hidden_dim": {
            "type": "categorical",
            "choices": [2048, 4096, 6144],
            "description": "Adapter intermediate dimension"
        },

        # Dropout rate
        "dropout_rate": {
            "type": "uniform",
            "low": 0.0,
            "high": 0.3,
            "description": "Dropout rate for regularization"
        },

        # Residual scale
        "residual_scale": {
            "type": "uniform",
            "low": 0.05,
            "high": 0.5,
            "description": "Scale factor for residual connections"
        },

        # Contrastive weight
        "contrastive_weight": {
            "type": "uniform",
            "low": 0.0,
            "high": 0.1,
            "description": "Weight for contrastive loss"
        },

        # Text Drift Loss Parameters (NEW)
        "text_drift_weight": {
            "type": "uniform",
            "low": 0.01,
            "high": 0.1,
            "description": "Weight for cosine similarity drift loss"
        },

        "text_drift_target_similarity": {
            "type": "uniform",
            "low": 0.7,
            "high": 0.9,
            "description": "Target cosine similarity for drift loss"
        },

        # Batch size (discrete choices)
        "batch_size": {
            "type": "categorical",
            "choices": [4, 8, 16],
            "description": "Training batch size"
        },

        # Gradient accumulation steps (dependent on batch size)
        "gradient_accumulation_steps": {
            "type": "categorical",
            "choices": [2, 4, 8],
            "description": "Gradient accumulation steps"
        }
    }


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters from the search space.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of suggested hyperparameters
    """
    search_space = get_search_space()
    params = {}

    for name, config in search_space.items():
        if config["type"] == "log_uniform":
            params[name] = trial.suggest_loguniform(
                name=name,
                low=config["low"],
                high=config["high"]
            )
        elif config["type"] == "uniform":
            params[name] = trial.suggest_uniform(
                name=name,
                low=config["low"],
                high=config["high"]
            )
        elif config["type"] == "categorical":
            params[name] = trial.suggest_categorical(
                name=name,
                choices=config["choices"]
            )
        elif config["type"] == "int":
            params[name] = trial.suggest_int(
                name=name,
                low=config["low"],
                high=config["high"]
            )

    return params


def validate_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and potentially adjust hyperparameters.

    Args:
        params: Raw hyperparameters from optimization

    Returns:
        Validated and adjusted hyperparameters
    """
    validated = params.copy()

    # Ensure effective batch size is reasonable
    batch_size = params["batch_size"]
    grad_accum = params["gradient_accumulation_steps"]
    effective_batch = batch_size * grad_accum

    # If effective batch size too large, reduce gradient accumulation
    if effective_batch > 64:
        validated["gradient_accumulation_steps"] = max(1, 64 // batch_size)
    elif effective_batch < 8:
        validated["gradient_accumulation_steps"] = max(2, 8 // batch_size)

    # Ensure hidden_dim is multiple of embedding_dim for efficiency
    embedding_dim = 2560  # Qwen3-Embedding-4B dimension
    hidden_dim = validated["hidden_dim"]
    if hidden_dim % embedding_dim != 0:
        # Round to nearest multiple
        remainder = hidden_dim % embedding_dim
        if remainder < embedding_dim // 2:
            validated["hidden_dim"] = hidden_dim - remainder
        else:
            validated["hidden_dim"] = hidden_dim + (embedding_dim - remainder)

    # Clamp contrastive weight to reasonable range
    validated["contrastive_weight"] = min(0.1, max(0.0, validated["contrastive_weight"]))

    # Ensure text drift weight is reasonable
    validated["text_drift_weight"] = min(0.1, max(0.01, validated["text_drift_weight"]))

    # Ensure target similarity is in valid cosine similarity range
    validated["text_drift_target_similarity"] = min(0.95, max(0.5, validated["text_drift_target_similarity"]))

    return validated


def get_sampler_config() -> Dict[str, Any]:
    """Get configuration for Optuna TPE sampler.

    Returns:
        Dictionary containing sampler configuration
    """
    return {
        "sampler_type": "tpe",
        "n_startup_trials": 10,  # Random trials before TPE
        "n_ei_candidates": 24,   # Number of candidates for Expected Improvement
        "multivariate": True,     # Enable multivariate TPE
        "group": True,            # Enable group TPE for better handling of conditional params
        "seed": 42,
    }


def get_pruner_config() -> Dict[str, Any]:
    """Get configuration for Optuna median pruner.

    Returns:
        Dictionary containing pruner configuration
    """
    return {
        "pruner_type": "median",
        "n_startup_trials": 5,      # Number of trials before pruning starts
        "n_warmup_steps": 2,        # Number of steps before pruning can occur (2-epoch constraint)
        "interval_steps": 1,        # Check pruning every epoch
        "direction": "maximize",    # Maximize BERTScore composite
    }


def create_study(
    study_name: str,
    direction: str = "maximize",
    storage_url: str = None,
    load_if_exists: bool = True
) -> optuna.Study:
    """Create and configure Optuna study for optimization.

    Args:
        study_name: Name of the study
        direction: Optimization direction ("maximize" or "minimize")
        storage_url: Optional database URL for persistence
        load_if_exists: Whether to load existing study

    Returns:
        Configured Optuna study
    """
    sampler_config = get_sampler_config()
    pruner_config = get_pruner_config()

    # Create TPE sampler
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=sampler_config["n_startup_trials"],
        n_ei_candidates=sampler_config["n_ei_candidates"],
        multivariate=sampler_config["multivariate"],
        group=sampler_config["group"],
        seed=sampler_config["seed"]
    )

    # Create median pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_config["n_startup_trials"],
        n_warmup_steps=pruner_config["n_warmup_steps"],
        interval_steps=pruner_config["interval_steps"]
    )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=load_if_exists
    )

    return study


def get_default_objective_weights() -> Dict[str, float]:
    """Get default weights for multi-objective optimization.

    Returns:
        Dictionary of metric weights
    """
    return {
        "bertscore_composite": 0.6,  # Primary objective
        "bertscore_f1": 0.2,          # F1 score
        "val_loss": -0.2              # Negative weight (minimize loss)
    }


def format_trial_params(params: Dict[str, Any]) -> str:
    """Format trial parameters for logging.

    Args:
        params: Trial parameters

    Returns:
        Formatted string representation
    """
    formatted = []

    # Order parameters for readability
    key_order = [
        "learning_rate", "hidden_dim", "batch_size", "gradient_accumulation_steps",
        "dropout_rate", "residual_scale", "contrastive_weight",
        "text_drift_weight", "text_drift_target_similarity"
    ]

    for key in key_order:
        if key in params:
            value = params[key]
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.6f}" if key == "learning_rate" else f"{key}: {value:.4f}")
            else:
                formatted.append(f"{key}: {value}")

    return ", ".join(formatted)