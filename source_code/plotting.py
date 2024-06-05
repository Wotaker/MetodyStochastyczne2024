import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor

# Set plotting style
sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100


def plot_samples(
        gpr: GaussianProcessRegressor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        mean: float,
        std: float,
        n_samples: int = 10,
        normalize: bool = False,
        title: str = 'Gaussian Process',
) -> None:
    gp_prior_samples = gpr.sample_y(X=x_train, n_samples=n_samples)

    fig, ax = plt.subplots()
    for i in range(n_samples):
        sns.lineplot(x=x_train[..., 0], y=gp_prior_samples[:, i], color=sns_c[1], alpha=0.2, ax=ax)
    if normalize:
        sns.lineplot(x=x_train[..., 0], y=(y_train[..., 0] - mean) / std, color=sns_c[0], label='y', ax=ax)
    else:
        sns.lineplot(x=x_train[..., 0], y=y_train[..., 0], color=sns_c[0], label='y', ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(title=title, xlabel='t');
    plt.show()


def plot_predictions(
        df: pd.DataFrame,
        n_train: int,
        r2_score_test: float,
        results_dir: str,
        verbose: bool = False
):
    # Plot the predictions
    fig, ax = plt.subplots()
    ax.fill_between(
        x=df['time'],
        y1=df['y_lwr'],
        y2=df['y_upr'],
        color=sns_c[2],
        alpha=0.15,
        label='$\mu \pm 2\sigma$'
    )
    sns.lineplot(x='time', y='CO2', data=df, color=sns_c[0], label='y_true', ax=ax)
    sns.lineplot(x='time', y='y_mean', data=df, color=sns_c[2], label='y_hat_mean', ax=ax)
    ax.axvline(n_train, color=sns_c[3], linestyle='--', label='train-test split')
    ax.legend()
    ax.set(title=f'Test R2 score: {r2_score_test:.5f}', xlabel='t', ylabel='');

    # Save figur and show
    plt.savefig(os.path.join(results_dir, 'predictions.pdf'), bbox_inches='tight')
    if verbose:
        plt.show()


def plot_errors(
        gpr: GaussianProcessRegressor,
        x_test: np.ndarray,
        y_test: np.ndarray,
        results_dir: str,
        verbose: bool = False
):
    # Calculate prediction errors
    errors = gpr.predict(x_test) - np.squeeze(y_test)
    errors = errors.flatten()
    errors_mean = errors.mean()
    errors_std = errors.std()

    # Plot prediction errors
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.regplot(x=y_test.flatten(), y=gpr.predict(x_test).flatten(), ax=ax[0])
    ax[0].plot(y_test.flatten(), y_test.flatten(), color=sns_c[3], linestyle='--', label='y=x')
    sns.histplot(data=errors, ax=ax[1], bins=25)
    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--', label=f'$\mu$')
    ax[1].axvline(x=errors_mean + 2 * errors_std, color=sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
    ax[1].axvline(x=errors_mean - 2 * errors_std, color=sns_c[4], linestyle='--')
    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
    ax[0].legend()
    ax[1].legend()
    ax[0].set(title='Gold vs Predictions (Test Set)', xlabel='y_test', ylabel='y_pred');
    ax[1].set(title='Errors', xlabel='error', ylabel=None);

    # Save figur and show
    plt.savefig(os.path.join(results_dir, 'errors.pdf'), bbox_inches='tight')
    if verbose:
        plt.show()
