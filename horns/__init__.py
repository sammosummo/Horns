from logging import getLogger
from multiprocessing import Pool
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from numba import njit, prange
from numba.core import config
from scipy import optimize
from scipy.stats import multivariate_normal
from scipy.stats.contingency import crosstab
from numba_stats import norm
from tqdm import tqdm

# ensure that numba is using the threading layer
config.THREADING_LAYER = "threadsafe"

# global constant for the maximum number of unique values in an ordinal variable
MAX_THRESHOLDS = 8
MT = MAX_THRESHOLDS


@njit(parallel=True, fastmath=True)
def simulate(data: np.ndarray, meth: str = "bootstrap") -> np.ndarray:
    """Generate a random dataset with the same shape as `data`.

    Args:
        data: Original data.
        meth: Method to use. Available methods are ``"normal"``, ``"uniform"``,
            ``"shuffle"``, and ``"bootstrap"``. ``"normal"`` will generate random data
            from a standard normal distribution. ``"uniform"`` will generate random data
            from a standard uniform distribution. ``"shuffle"`` will shuffle the data in
            each column, maintaining the original univariate distribution of each
            variable. ``"bootstrap"`` will generate new columns by sampling with
            replacement from each column. The default is ``"bootstrap"``.

    Returns:
        Random data in an array with the same shape as ``data``.

    Raises:
        ValueError: If the method is not recognised.

    """
    n, m = data.shape
    out = np.empty((n, m), dtype=data.dtype)

    if meth == "normal":
        for i in prange(n):
            for j in prange(m):
                out[i, j] = np.random.normal()

    elif meth == "uniform":
        for i in prange(n):
            for j in prange(m):
                out[i, j] = np.random.uniform()

    elif meth == "shuffle":
        for i in prange(m):
            out[:, i] = np.random.permutation(data[:, i])

    elif meth == "bootstrap":
        for i in prange(m):
            # bootstrapping is considerably faster than shuffling and is more widely
            # applicable than normal or uniform
            out[:, i] = np.random.choice(data[:, i], size=n, replace=True)

    else:
        raise ValueError(f"Method {meth} not recognised.")

    return out


@njit(parallel=True, fastmath=True)
def simulate_all(data: np.ndarray, s: int, method: str) -> np.ndarray:
    """Generate ``s`` random datasets each with the same shape as ``data``.

    Args:
        data: Original data.
        s: Number of datasets to generate.
        method: Method to use. See :func:`simulate`.

    Returns:
        Random datasets in an array with shape `(s, n, m)`, where `s` is the number of
            datasets, `n` is the number of observations per variable, and `m` is the
            number of variables.

    """
    n, m = data.shape
    out = np.empty((s, n, m), dtype=data.dtype)

    for i in prange(s):
        out[i] = simulate(data, method)

    return out


@njit(fastmath=True)
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the Pearson correlation coefficient between two arrays.

    This function calls ``np.corrcoef`` and extracts the correlation coefficient from
    the off-diagonal of the resulting matrix. Isolated as a separate function to allow
    JIT compilation with Numba.

    Args:
        x: First array of data. Both ``x`` and ``y`` should be continuous or ordinal
            with 8 or more unique values, otherwise this is most likely the wrong type
            of correlation.
        y: Second array of data.

    Returns:
        Pearson correlation coefficient.

    """
    return np.corrcoef(x, y)[0, 1].item()


@njit(fastmath=True)
def pearson_matrix(data: np.ndarray) -> np.ndarray:
    """Calculate the Pearson correlation matrix of a real or simulated dataset.

    Args:
        data: Data. All columns should be continuous or ordinal with more than 8 unique
        values, otherwise this is most likely the wrong function to call.

    Returns:
        Correlation matrix.

    """
    return np.corrcoef(data, rowvar=False)


@njit(parallel=True, fastmath=True)
def pearson_matrices(a: np.ndarray) -> np.ndarray:
    """Calculate the Pearson correlation matrices for all datasets.

    Args:
        a: Three-dimensional array of datasets.

    Returns:
        Correlation matrix.

    """
    s, n, m = a.shape
    out = np.empty((s, m, m), dtype=np.float64)

    for i in prange(s):
        out[i] = pearson_matrix(a[i])

    return out


@njit(parallel=True, fastmath=True)
def phi(x: np.ndarray) -> np.ndarray:
    """Calculate the density of the standard normal distribution at the given values.

    JIT-compiled version of the function from ``numba-stats``.

    Args:
        x: values.

    Returns:
        Density.

    """
    return norm.pdf(x, 0.0, 1.0)  # noqa


@njit(parallel=True, fastmath=True)
def quantile(p: np.ndarray) -> np.ndarray:
    """Quantile function of the standard normal distribution.

    JIT-compiled version of the function from ``numba-stats``.

    Args:
        p: Probabilities.

    Returns:
        Quantiles.

    """
    return norm.ppf(p, 0.0, 1.0)  # noqa


@njit(fastmath=True)
def polyserial(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the polyserial correlation coefficient between two arrays using the
    ad-hoc method described by `Olsson et al. (1982)`_.

    Args:
        x: Array of ordinal data with 8 or fewer consecutive integer unique values.
            This function doesn't check the data for correctness; this should be done
            beforehand.
        y: Array of continuous data. ``y`` should be continuous or ordinal with more
            than 8 unique values.

    Returns:
        Polyserial correlation coefficient.

    .. _Olsson et al. (1982): https://link.springer.com/article/10.1007/BF02294164

    """
    xl = np.unique(x)
    xt = np.sum(x == xl[:, None], axis=1)
    p = np.cumsum(xt)[:-1] / np.sum(xt)
    tau = quantile(p)
    return pearson(x, y) * np.std(x) / np.sum(phi(tau))


@njit(parallel=True, fastmath=True)
def _crosstab(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create a contingency table from two arrays.

    This appears to be slightly slower than scipy's :func:`crosstab` even when JITed.

    Args:
        x: First array of data.
        y: Second array of data.

    Returns:
        Contingency table.

    """
    nx = np.unique(x)
    ny = np.unique(y)
    out = np.zeros((nx.size, ny.size), dtype=np.float64)

    for i in prange(nx.size):
        for j in prange(ny.size):
            out[i, j] = np.sum((x == nx[i]) & (y == ny[j]))

    return out


@njit(parallel=True, fastmath=True)
def _pxy(p: np.ndarray) -> np.ndarray:
    """Apply the quantile function to the probabilities padded with -23 and 23.

    Args:
        p: Probabilities.

    Returns:
        Marginal probabilities.

    Raises:
        AssertionError: If the resulting array is not strictly increasing.

    """
    q = np.empty(p.size + 2, dtype=np.float64)
    q[0] = -23.0
    q[1:-1] = quantile(np.cumsum(p))
    q[-1] = 23.0
    assert np.all(np.diff(q) > 0)
    return q


def polychoric(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the polychoric correlation coefficient between two arrays using the
    two-step method described by `Martinson and Hamdan (1972)`_.

    Notes:
        Code was inspired by `RyStats`_.

    Args:
        x: First array of data. Both ``x`` and ``y`` should be ordinal with 8 or fewer
            unique values, otherwise this is most likely the wrong type of correlation.
        y: Second array of ordinal data. See ``x``.

    Returns:
        Polychoric correlation coefficient.

    .. _Martinson and Hamdan (1972): https://doi.org/10.1080/00949657208810003
    .. _RyStats: https://github.com/eribean/RyStats

    """
    t = crosstab(x, y).count
    tt = np.sum(t)
    px = np.sum(t, axis=1)[:-1] / tt
    qx = _pxy(px)
    py = np.sum(t, axis=0)[:-1] / tt
    qy = _pxy(py)
    return optimize.fminbound(_nll, -0.999, 0.999, args=(t, qx, qy))  # noqa


def _nll(rho: float, t: np.ndarray, px: np.ndarray, py: np.ndarray) -> float:
    """Calculate the negative log-likelihood of the polychoric model.

    Uses the bivariate normal log cdf from ``scipy.stats``. Private function to be
    called by ``polychoric``.

    Args:
        rho: Polychoric correlation coefficient.
        t: Contingency table.
        px: Thresholds for x.
        py: Thresholds for y.

    Returns:
        Negative log-likelihood.

    """
    out = np.empty_like(t, dtype=float)
    dist = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])

    for i in prange(1, px.size):
        for j in prange(1, py.size):
            ll = dist.logcdf((px[i], py[j]), lower_limit=(px[i - 1], py[j - 1]))
            out[i - 1, j - 1] = ll

    return -1 * np.sum(t * out)


def polychoric_matrix(data: np.ndarray) -> np.ndarray:
    """Calculate the polychoric correlation matrix.

    Args:
        data: Data. All columns should be ordinal with 8 or fewer unique values.

    Returns:
        Correlation matrix.

    """
    n, m = data.shape
    out = np.eye(m)

    for i in range(m):
        for j in range(i + 1, m):
            out[i, j] = out[j, i] = polychoric(data[:, i], data[:, j])

    return out


def polychoric_matrices(a: np.ndarray) -> np.ndarray:
    """Calculate the polychoric correlation matrices for all datasets.

    Args:
        a: Three-dimensional array of datasets.

    Returns:
        Correlation matrix.

    """
    s, n, m = a.shape
    out = np.empty((s, m, m), dtype=np.float64)

    try:
        with Pool() as pool:
            results = pool.map(polychoric_matrix, a)

        for i in range(s):
            out[i] = results[i]

    except AssertionError:
        for i in tqdm(list(range(s))):
            out[i] = polychoric_matrix(a[i])

    return out


def het_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Returns the Pearson, polyserial, or polychoric correlation coefficient depending
    on the number of unique values in the input arrays.

    Args:
        x: First array of data.
        y: Second array of data.

    Returns:
        Correlation coefficient and type.

    """
    if np.unique(x).size > MT and np.unique(y).size > MT:
        return pearson(x, y)

    elif np.unique(x).size <= MT and np.unique(y).size <= MT:
        return polychoric(x, y)

    # PyCharm wants me to simplify this condition, but I find it easy to read this way
    elif np.unique(x).size <= MT and np.unique(y).size > MT:  # noqa
        return polyserial(x, y)
    
    elif np.unique(x).size > MT and np.unique(y).size <= MT: # noqa
        return polyserial(y, x)
    
    else:
        # this should never happen (probably)
        raise ValueError("Correlation type could not be determined.")


@njit(parallel=True, fastmath=True)
def _transform(data: np.ndarray) -> np.ndarray:
    """Transform ordinal data to consecutive integers.

    Olsson et al.'s polyserial approximation requires this.

    """
    out = np.empty_like(data, dtype=np.float64)

    for i in range(data.shape[1]):
        x = data[:, i]
        ux = np.unique(x)
        if ux.size <= MT:
            x2 = np.empty_like(x, dtype=np.float64)
            for j in range(ux.size):
                x2[x == ux[j]] = j
            out[:, i] = x2
        else:
            out[:, i] = x

    return out


def het_corr_matrix(data: np.ndarray) -> np.ndarray:
    """Calculate the correlation matrix containing Pearson, polyserial, and/or
    polychoric correlation coefficients.

    This function automatically transforms ordinal data to consecutive integers.

    Args:
        data: Data.

    Returns:
        Correlation matrix.

    """
    data = _transform(data)

    n, m = data.shape
    out = np.eye(m)

    for i in range(m):
        for j in range(i + 1, m):
            out[i, j] = out[j, i] = het_corr(data[:, i], data[:, j])

    return out


def correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Calculate the correlation matrix of a real or simulated dataset.

    Args:
        data: Data.

    Returns:
        Correlation matrix.

    """
    n = []
    for i in range(data.shape[1]):
        n.append(np.unique(data[:, i]).size)
    
    if all(x > MT for x in n):
        return pearson_matrix(data)
    
    elif all(x <= MT for x in n):
        return polychoric_matrix(data)
    
    else:
        return het_corr_matrix(data)


def correlation_matrices(a: np.ndarray) -> np.ndarray:
    """Calculate the correlation matrices for all datasets.

    Args:
        a: Three-dimensional array of datasets.

    Returns:
        Correlation matrix.

    """
    ns = []
    for i in range(a.shape[2]):
        ns.append(np.unique(a[:, :, i]).size)

    if all(x > MT for x in ns):
        return pearson_matrices(a)

    elif all(x <= MT for x in ns):
        return polychoric_matrices(a)

    else:
        s, n, m = a.shape
        out = np.empty((s, m, m), dtype=np.float64)

        try:
            with Pool() as pool:
                results = pool.map(correlation_matrix, a)

            for i in range(s):
                out[i] = results[i]

        except AssertionError:
            for i in tqdm(list(range(s))):
                out[i] = correlation_matrix(a[i])

    return out


@njit(parallel=True, fastmath=True)
def eigenvalues(a: np.ndarray, at: str) -> np.ndarray:
    """Compute the eigenvalues of a 2D array.

    Args:
        a: Array. For the output of this function to make sense, ``a`` must be a
            real-valued symmetric (correlation) matrix
        at: Type of analysis being performed. Either "pca" or "fa". This is
            important because the eigenvalues will be calculated differently depending
            on the type of analysis; see `Dinno (2009)`_.

    Returns:
        Eigenvalues in an array.

    .. _Dinno (2009): https://archives.pdx.edu/ds/psu/10527

    """
    if str(at) == "pca":
        return np.linalg.eigvals(a)

    elif str(at) == "fa":
        aplus = np.linalg.pinv(a)
        d = np.diag(np.diag(aplus))
        p = np.linalg.pinv(d)
        return np.linalg.eigvals(a - p)

    else:
        raise ValueError(f"Analysis type {at} not recognised.")


@njit(parallel=True, fastmath=True)
def all_eigenvalues(a: np.ndarray, at: str) -> np.ndarray:
    """Compute eigenvalues of all datasets.

    Args:
        a: Three-dimensional array of datasets.
        at: Type of analysis.

    Returns:
        Eigenvalues.

    """
    s, n, m = a.shape
    out = np.empty((s, m), dtype=np.float64)

    for i in prange(s):
        out[i] = eigenvalues(a[i], at)

    return out


class _DummyLogger:
    def debug(self, *args, **kwargs):
        pass


def parallel_analysis(
    data: np.ndarray,
    simulations: int = int(1e4),
    randomisation_method: str = "bootstrap",
    analysis_type: str = "pca",
    quartile: float = 0.95,
    full_output: bool = False,
    verbose: bool = False,
) -> dict[str, Any] | int:
    """Perform Horn's parallel analysis.

    Parallel analysis involves the following steps: (1) calculate the eigenvalues of the
    correlation matrix of the original dataset; (2) generate a random dataset with the
    same shape as the original dataset but with no underlying correlation structure;
    (3) calculate the eigenvalues of the random dataset; (4) repeat steps 2 and 3 many
    times; (5) calculate the `q`th quantile of the distribution of each random-data
    eigenvalue; (6) apply the decision rule to determine the number of components or
    factors to retain. The decision can be expressed as "count the number of original
    eigenvalues that are greater than their corresponding quantile until we encounter
    the first eigenvalue that is not greater than its quantile".

    Args:
        data: An NumPy array with shape ``(n, m)``, where ``n`` is the number of
            observations per variable and ``m`` is the number of variables. Currently,
            we only support complete datasets, so any missing values must be handled
            before calling this function. It is probably best to delete all rows with
            missing values.
        simulations: The default is ``10000``. You may wish to reduce this number if
            you are working with a large dataset, on a slow machine, or working with
            ordinal variables.
        randomisation_method: Must be ``"normal"``, ``"uniform"``, ``"shuffle"``, or
        ``"bootstrap"``. If ``data`` contains ordinal variables, you must use
        ``"shuffle"`` or ``"bootstrap"`` to ensure that the random data are also
        ordinal. Defaults to ``"bootstrap"``.
        analysis_type: Either ``"pca"`` (the default) or ``"fa"``.
        quartile: Must be a float in the range ``0<q<1``. Defaults to ``0.95``.
        full_output: Whether to return all of the results in a dictionary or just the
            integer representing the number of components/factors to retain. Defaults to
            ``False``.
        verbose: If ``True``, debugging information will sent to Python's logging
            system. Defaults to ``False``.

    Returns:
        If ``return_all`` is ``False``, the number of components or factors to retain.
        Otherwise, a dictionary containing various details from the analysis.

    """
    if verbose:
        logger = getLogger(__name__)
    else:
        logger = _DummyLogger()

    logger.debug("Calculating the correlation matrix of the original dataset.")
    mat = correlation_matrix(data)

    logger.debug("Calculating the observed eigenvalues.")
    eig = eigenvalues(mat, analysis_type)

    logger.debug("Generating random datasets.")
    rands = simulate_all(data, simulations, randomisation_method)

    logger.debug("Calculating the correlation matrices of the random datasets.")
    mats = correlation_matrices(rands)

    logger.debug("Calculating the eigenvalues of the random correlation matrices.")
    eigs = all_eigenvalues(mats, analysis_type)

    logger.debug("Calculating criteria.")
    crit = np.quantile(eigs, quartile, axis=0)

    logger.debug("Applying the decision rule.")
    acc = eig > crit
    factors = np.where(~acc)[0][0].item()

    if full_output:
        logger.debug("Returning full output.")
        results = {
            "correlation_matrix": mat,
            "eigenvalues": eig,
            "random_datasets": rands,
            "random_correlation_matrices": mats,
            "random_eigenvalues": eigs,
            "criteria": crit,
            "accepted": acc,
            "factors": factors,
        }

        logger.debug("Generating the scree plot.")
        fig, ax = plt.subplots()
        ax.set_xlabel(f"{'Factor' if analysis_type == 'fa' else 'Component'} number")
        ax.set_ylabel("Eigenvalue")
        x = np.arange(1, eig.size + 1)
        ax.plot(x, eig, label="Observed", color="black", marker="o")
        ax.plot(x, crit, label=f"Threshold", color="red")
        ax.legend()
        results["figure"] = fig

        return results

    return factors
