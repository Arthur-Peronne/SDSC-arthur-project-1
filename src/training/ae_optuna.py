# src/training/ae_optuna.py
"""
Functions for ptuna hyperparameter optimisation for AE3dFCDeep used in run_ae_hyperparam.py

Each trial trains AE3dFCDeep with early stopping and returns
the best validation loss. Results are persisted in SQLite so
the study can be interrupted and resumed at any time.
"""

import optuna
import optuna.trial as optuna_trial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from src.config import TEMPODATA_FOLDER
from src.training import ae_training as aet


def objective(trial: optuna.Trial, config: dict) -> float:

    lr           = trial.suggest_float("lr",           1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0,  0.3)
    noise_std    = trial.suggest_float("noise_std",    0.0,  0.005)
    patience     = trial.suggest_int(  "patience",     20,   80)

    n_train_images = (
        (config["n_development"] - config["n_validation"])
        * (2 if config["use_both_frames"] else 1)
    )
    simulation_name = (
        f"{config['model_name']}_{n_train_images}patients"
        f"_{config['splitname']}_{config['latent_dim']}dims"
    )
    experiment_name = f"optuna_{config['study_name']}_trial{trial.number}"

    train_dataset, validation_dataset, _, _ = aet.ae_getdataset(
        n_patients=config["n_development"],
        validation=True,
        n_validation=config["n_validation"],
        imagesource="registered_frames",
        use_both_frames=config["use_both_frames"],
        recalculateXvector=False,
    )

    try:
        _, best_epoch, loss_history = aet.ae_training_early_stopping(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            simulation_name=simulation_name,
            model_name=config["model_name"],
            latent_dimensions=config["latent_dim"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            lr=lr,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            noise_std=noise_std,
            patience=patience,
            patience_scheduler=None,
            recalculateAE=True,
            experiment_name=experiment_name,
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()

    best_val_loss = min(loss_history["validation"])
    print(
        f"Trial {trial.number:>3} | val_loss={best_val_loss:.6f} | "
        f"epoch={best_epoch:>4} | lr={lr:.2e} | wd={weight_decay:.2e} | "
        f"drop={dropout_rate:.2f} | noise={noise_std:.3f} | patience={patience}"
    )
    trial.set_user_attr("best_epoch", best_epoch)
    return best_val_loss


def run_optuna(config: dict) -> optuna.Study:

    config["db_path"].parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=config["study_name"],
        storage=f"sqlite:///{config['db_path']}",
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        study.enqueue_trial({
            "lr":           1e-5,
            "weight_decay": 1e-7,
            "dropout_rate": 0.0,
            "noise_std":    0.0,
            "patience":     40,
        })

    # Check how many trials have been done
    n_done = len([t for t in study.trials 
              if t.state == optuna_trial.TrialState.COMPLETE])
    n_remaining = max(0, config["n_trials"] - n_done)
    
    print(f"Trials déjà complétés : {n_done} / {config['n_trials']}")
    print(f"Trials restants : {n_remaining}")

    # functools.partial to give config to objective
    import functools
    obj_with_config = functools.partial(objective, config=config)

    if n_remaining > 0:
        study.optimize(obj_with_config, n_trials=n_remaining, show_progress_bar=True)

    save_optuna_results(study, config["db_path"])

    return study

def save_optuna_results(study: optuna.Study, db_path: Path) -> None:
    """
    Save a human-readable summary of the Optuna study to a .txt file.
    """
    output_path = db_path.parent / f"{study.study_name}_results.txt"

    lines = []
    lines.append("=" * 60)
    lines.append(f"Optuna Study : {study.study_name}")
    lines.append(f"Trials completed : {len(study.trials)}")
    lines.append(f"Best val loss    : {study.best_value:.6f}")
    lines.append("")
    lines.append("Best hyperparameters :")
    for k, v in study.best_params.items():
        lines.append(f"  {k:>15} = {v}")
    lines.append("")

    # Hyperparameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        lines.append("Hyperparameter importance :")
        for k, v in importance.items():
            bar = "█" * int(v * 30)
            lines.append(f"  {k:>15} : {v:.3f}  {bar}")
        lines.append("")
    except Exception:
        lines.append("(importance non disponible — trop peu de trials)")
        lines.append("")

    # All trials table
    lines.append(f"{'Trial':>6} {'Val loss':>12} {'lr':>10} {'wd':>10} "
                 f"{'dropout':>8} {'noise':>8} {'patience':>9} {'epoch':>6}")
    lines.append("-" * 75)
    for t in sorted(study.trials, key=lambda t: t.value or float("inf")):
        if t.value is None:
            continue
        p = t.params
        lines.append(
            f"{t.number:>6} {t.value:>12.6f} "
            f"{p.get('lr', 0):>10.2e} "
            f"{p.get('weight_decay', 0):>10.2e} "
            f"{p.get('dropout_rate', 0):>8.2f} "
            f"{p.get('noise_std', 0):>8.3f} "
            f"{p.get('patience', 0):>9} "
            f"{t.user_attrs.get('best_epoch', '?'):>6}"
        )
    lines.append("=" * 60)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Results saved to {output_path}")


def load_and_plot(config: dict) -> None:
    """
    Load an existing Optuna study from SQLite and plot results.
    Use this to regenerate plots without rerunning the optimization.
    """
    study = optuna.load_study(
        study_name=config["study_name"],
        storage=f"sqlite:///{config['db_path']}",
    )
    plot_optuna_results(study, config["db_path"])


def plot_optuna_results(study: optuna.Study, db_path: Path) -> None:
    """
    Plot optimization history and hyperparameter evolution across trials.
    Saves two figures in the same folder as the SQLite DB.
    """

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.number)

    trial_nums = [t.number for t in completed]
    val_losses = [t.value for t in completed]
    best_idx   = val_losses.index(min(val_losses))
    best_trial = trial_nums[best_idx]

    # ── Best so far curve ─────────────────────────────────────────
    best_so_far = []
    running_min = float("inf")
    for v in val_losses:
        running_min = min(running_min, v)
        best_so_far.append(running_min)

    output_dir = db_path.parent

    # ══════════════════════════════════════════════════════════════
    # Plot 1 — Optimization history
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(trial_nums, val_losses,
            color="#B4B2A9", linewidth=1, marker="o", markersize=3,
            label="Val loss")
    # ax.plot(trial_nums, best_so_far,
    #         color="#534AB7", linewidth=2, drawstyle="steps-post",
    #         label="Best so far")
    ax.axvline(x=best_trial, color="#E24B4A", linewidth=1.2,
               linestyle="--", label=f"Best trial ({best_trial})")
    ax.scatter([best_trial], [min(val_losses)],
               color="#E24B4A", s=60, zorder=5)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Val loss")
    ax.set_title(f"Optuna — optimization history\n{study.study_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    sorted_losses = sorted(val_losses, reverse=True)
    ax.set_ylim(bottom=min(val_losses) * 0.95, top=sorted_losses[1]*1.02)

    fig.tight_layout()
    fig.savefig(output_dir / f"{study.study_name}_history.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / f'{study.study_name}_history.png'}")

   
    # ══════════════════════════════════════════════════════════════
    # Plot 2 — One subplot per hyperparameter
    # ══════════════════════════════════════════════════════════════
    hp_config = {
        "lr":           ("#534AB7", "lr"),
        "weight_decay": ("#1D9E75", "weight decay"),
        "dropout_rate": ("#D85A30", "dropout"),
        "noise_std":    ("#BA7517", "noise"),
        "patience":     ("#378ADD", "patience"),
    }

    n_hp = len(hp_config)
    fig, axes = plt.subplots(n_hp, 1, figsize=(10, 3 * n_hp), sharex=True)

    # Val loss normalized
    vl_min, vl_max = min(val_losses), max(val_losses)
    vl_norm = [(v - vl_min) / (vl_max - vl_min) for v in val_losses]

    for ax, (hp_name, (color, label)) in zip(axes, hp_config.items()):

        # Val loss in background
        ax_vl = ax.twinx()
        ax_vl.plot(trial_nums, val_losses,
                   color="#B4B2A9", linewidth=0.8, linestyle="--", alpha=0.5)
        sorted_losses = sorted(val_losses, reverse=True)
        ax_vl.set_ylim(bottom=min(val_losses) * 0.95, top=sorted_losses[1]*1.02)
        ax_vl.set_ylabel("val loss", fontsize=8, color="#B4B2A9")
        ax_vl.tick_params(axis="y", labelcolor="#B4B2A9", labelsize=7)

        # Hyperparameter
        values = [t.params.get(hp_name) for t in completed]
        ax.plot(trial_nums, values,
                color=color, linewidth=1.5, marker="o", markersize=3)
        ax.axvline(x=best_trial, color="#E24B4A", linewidth=1.2, linestyle="--")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Scientific notation and log scale
        if hp_name in ("lr", "weight_decay"):
            ax.yaxis.set_major_formatter(
                plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_yscale("log")

    axes[-1].set_xlabel("Trial number")
    fig.suptitle(f"Optuna — hyperparameter evolution\n{study.study_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / f"{study.study_name}_hyperparams.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / f'{study.study_name}_hyperparams.png'}")