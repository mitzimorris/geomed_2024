# import all libraries used in this notebook
import numpy as np
import pandas as pd
import plotnine as p9

def upper_corr_matrix_to_df(draws: np.ndarray) -> pd.DataFrame:
    """
    Compute correlation for a parameter vector.
    Given a 2-d array of draws (rows = draws, columns = variables),
    compute the correlation matrix for the upper triangle.
    - param draws: np.ndarray, N x 1 draws for a single parameter
    - returns a DataFrame ['Var1', 'Var2', 'Correlation']
    """
    cor_mat = np.corrcoef(draws, rowvar=False)

    # Mask the lower triangle (including diagonal)
    upper_cor = cor_mat.copy()
    lower_triangle_indices = np.tril_indices_from(upper_cor)
    upper_cor[lower_triangle_indices] = np.nan

    # Convert to DataFrame and melt
    df_cor = pd.DataFrame(upper_cor)
    melted = df_cor.reset_index().melt(
        id_vars='index',
        var_name='Var2',
        value_name='Correlation'
    )
    melted.rename(columns={'index': 'Var1'}, inplace=True)
    melted = melted.dropna()
    return melted

def plot_icar_corr_matrix(plot_df: pd.DataFrame, title: str, size: tuple[int, int]) -> p9.ggplot:
    p = (
        p9.ggplot(plot_df, p9.aes(x='Var1', y='Var2', fill='Correlation'))
        + p9.geom_tile()
        + p9.scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0)
        + p9.theme_minimal()
        + p9.theme(
            figure_size=size,
            axis_text_x=p9.element_blank(),
            axis_text_y=p9.element_blank())
        + p9.ylab('') + p9.xlab('')
        + p9.ggtitle(title)
    )
    return p


def ppc_y_yrep_overlay(
        y_rep: np.ndarray,
        y: pd.Series,
        title: str) -> p9.ggplot:
    y_rep_median = np.median(y_rep, axis=0)
    y_rep_lower = np.percentile(y_rep, 2.5, axis=0)
    y_rep_upper = np.percentile(y_rep, 97.5, axis=0)

    df_plot = pd.DataFrame({
        'obs_id': np.arange(y_rep.shape[1]),
        'y': y,
        'y_rep_median': y_rep_median,
        'y_rep_lower': y_rep_lower,
        'y_rep_upper': y_rep_upper
        })

    # Sort by y, reindex
    df_plot_sorted = df_plot.sort_values(by='y').reset_index(drop=True)
    df_plot_sorted['sorted_index'] = np.arange(len(df_plot_sorted))

    p = (
        p9.ggplot(df_plot_sorted, p9.aes(x='sorted_index'))
        + p9.geom_point(p9.aes(y='y'), color='darkblue', size=0.25)
        + p9.geom_line(p9.aes(y='y_rep_median'), color='orange', alpha=0.8)
        + p9.geom_ribbon(p9.aes(ymin='y_rep_lower', ymax='y_rep_upper'),
                                fill='grey', alpha=0.2)
        + p9.theme(figure_size=(10, 5), axis_text_x=p9.element_blank())
        + p9.ylab('y_rep') + p9.xlab('y')
        + p9.ggtitle(title)
        )
    return(p)

def ppc_central_interval(y_rep: np.ndarray, y: pd.Series) -> str:
    # Compute the 25th and 75th percentiles per observation
    q25 = np.percentile(y_rep, 25, axis=0)
    q75 = np.percentile(y_rep, 75, axis=0)

    # Count how many observed values fall within the central 50% interval
    within_50 = np.sum((y >= q25) & (y <= q75))

    return((
        f"y total: {y_rep.shape[1]}, "
        f"ct y is within y_rep central 50% interval: {within_50}, "
        f"pct: {100 * within_50 / y_rep.shape[1]}"))
