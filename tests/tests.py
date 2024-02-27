from unittest import TestCase


class Tests(TestCase):

    def test_requirements(self):
        """Test that all required packages are installed."""
        import numpy as np  # noqa
        import matplotlib.pyplot as plt  # noqa
        from matplotlib.figure import Figure  # noqa
        import numba  # noqa
        import numba_stats  # noqa
        import scipy  # noqa
        import tqdm  # noqa

    def test_pearson(self):
        """Test that pearson returns a single (correct) value."""
        import numpy as np
        from horns import pearson

        mu = [0, 0]
        cov = np.array([[1, 0.5], [0.5, 1]])
        x = np.random.multivariate_normal(mu, cov, 100000)

        r = pearson(x[:, 0], x[:, 1])
        self.assertIsInstance(r, float)
        self.assertAlmostEqual(r, 0.5, 1)  # noqa

    def test_pearson_matrix(self):
        """Test that pearson_matrix returns a symmetric matrix."""
        import numpy as np
        from horns import pearson_matrix
        from scipy.linalg import issymmetric

        data = np.random.rand(100, 100)
        out = pearson_matrix(data)
        self.assertEqual(out.shape, (100, 100))
        self.assertTrue(issymmetric(out))

    def test_pearson_matrices(self):
        import numpy as np
        from horns import pearson_matrices
        from scipy.linalg import issymmetric

        a = np.random.rand(100, 100, 100)
        out = pearson_matrices(a)
        self.assertEqual(out.shape, (100, 100, 100))
        for i in range(100):
            self.assertTrue(issymmetric(out[i]))

    def test_rand_numba_normal(self):
        """Normal random data generation."""
        import numpy as np
        from horns import simulate

        for i in range(100):
            data = np.random.rand(100, 100)
            out = simulate(data, "normal")
            if np.any(np.isnan(out)):
                raise ValueError("NaNs detected.")
            self.assertAlmostEqual(out.mean(), 0, 1)
            self.assertAlmostEqual(out.std(), 1, 1)

    def test_rand_numba_uniform(self):
        """Uniform random data generation."""
        import numpy as np
        from horns import simulate

        for i in range(100):
            data = np.random.rand(100, 100)
            out = simulate(data, "uniform")
            if np.any(np.isnan(out)):
                raise ValueError("NaNs detected.")
            self.assertAlmostEqual(out.mean(), 0.5, 1)
            self.assertAlmostEqual(out.std(), 1 / (12**0.5), 1)

    def test_rand_numba_shuffle(self):
        """Shuffle random data generation."""
        import numpy as np
        from horns import simulate

        for i in range(100):
            data = np.random.rand(100, 100)
            out = simulate(data, "shuffle")
            if np.any(np.isnan(out)):
                raise ValueError("NaNs detected.")
            self.assertFalse(np.all(data == out))
            self.assertAlmostEqual(data.sum(), out.sum())

    def test_rand_numba_bootstrap(self):
        """Bootstrap random data generation."""
        import numpy as np
        from horns import simulate

        for i in range(100):
            data = np.random.rand(100, 100)
            out = simulate(data, "bootstrap")
            if np.any(np.isnan(out)):
                raise ValueError("NaNs detected.")
            self.assertFalse(np.all(data == out))

    def test_simulate_all(self):
        """Test that simulate returns the correct shape."""
        import numpy as np
        from horns import simulate_all

        for method in ["normal", "uniform", "shuffle", "bootstrap"]:
            data = np.random.rand(100, 100)
            out = simulate_all(data, 100, method)
            self.assertEqual(out.shape, (100, 100, 100))

    @staticmethod
    def _data(rho):
        import numpy as np

        mean = [0, 0]
        cov = np.array([[1, rho], [rho, 1]])
        data = np.random.multivariate_normal(mean, cov, 10000)
        x = data[:, 0]
        y = data[:, 1]
        return x, y

    def test_polyserial(self):
        import numpy as np
        from horns import polyserial, pearson

        for rho in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]:
            x, y = self._data(rho)
            true = pearson(x, y)
            x = np.digitize(x, bins=np.quantile(x, [0.5])).astype(float)
            r = polyserial(x, y)
            self.assertAlmostEqual(true, r, 1)

    def test_polychoric(self):
        import numpy as np
        from horns import polychoric, pearson

        for rho in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]:
            x, y = self._data(rho)
            true = pearson(x, y)
            x = np.digitize(x, bins=np.quantile(x, [0.1, 0.5])).astype(float)
            y = np.digitize(y, bins=np.quantile(y, [0.25, 0.5, 0.75])).astype(float)
            r = polychoric(x, y)
            self.assertAlmostEqual(true, r, 1)
            r = polychoric(y, x)
            self.assertAlmostEqual(true, r, 1)

    def test_polychoric_matrix(self):
        import numpy as np
        from horns import polychoric_matrix
        from time import time

        data = np.random.rand(1000, 10)
        data = np.digitize(data, bins=np.quantile(data, [0.1, 0.5, 0.9])).astype(float)
        t = time()
        out = polychoric_matrix(data)
        print(time() - t, "seconds")
        self.assertTrue(np.allclose(out, out.T))

    def test_polychoric_matrices(self):
        import numpy as np
        from horns import polychoric_matrices
        from scipy.linalg import issymmetric
        from time import time

        a = np.random.rand(100, 100, 10)
        a = np.digitize(a, bins=np.quantile(a, [0.1, 0.5, 0.9]))
        now = time()
        out = polychoric_matrices(a)
        print(time() - now, "seconds")
        for i in range(100):
            self.assertTrue(issymmetric(out[i]))

    def test_het_corr(self):
        import numpy as np
        from horns import het_corr, pearson, polyserial, polychoric

        x = np.random.rand(1000)
        y = np.random.rand(1000)
        out = het_corr(x, y)
        self.assertTrue(np.allclose(out, pearson(x, y)))

        y = np.digitize(y, bins=np.quantile(y, [0.1, 0.5, 0.9])).astype(float)
        out = het_corr(x, y)
        self.assertTrue(np.allclose(out, polyserial(y, x)))

        out = het_corr(y, x)
        self.assertTrue(np.allclose(out, polyserial(y, x)))

        x = np.digitize(x, bins=np.quantile(x, [0.1, 0.5, 0.9])).astype(float)
        out = het_corr(x, y)
        self.assertTrue(np.allclose(out, polychoric(x, y)))

    def test_het_corr_matrix(self):
        import numpy as np
        from horns import het_corr_matrix

        data = np.random.rand(1000, 10)

        out = het_corr_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

        data[:, 5] = np.digitize(
            data[:, 5], bins=np.quantile(data[:, 5], [0.1, 0.5, 0.9])
        ).astype(float)
        out = het_corr_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

        data[:, 5:] = np.digitize(
            data[:, 5:], bins=np.quantile(data[:, 5:], [0.1, 0.5, 0.9])
        ).astype(float)
        out = het_corr_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

        data = np.digitize(data, bins=np.quantile(data, [0.1, 0.5, 0.9])).astype(float)
        out = het_corr_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

    def test_correlation_matrix(self):
        import numpy as np
        from horns import correlation_matrix

        data = np.random.rand(1000, 10)

        out = correlation_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

        data[:, 5] = np.digitize(
            data[:, 5], bins=np.quantile(data, [0.1, 0.5, 0.9])
        ).astype(float)
        out = correlation_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

        data[:, 5:] = np.digitize(
            data[:, 5:], bins=np.quantile(data, [0.1, 0.5, 0.9])
        ).astype(float)
        out = correlation_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

        data = np.digitize(data, bins=np.quantile(data, [0.1, 0.5, 0.9])).astype(float)
        out = correlation_matrix(data)
        self.assertTrue(np.allclose(out, out.T))

    def test_correlation_matrices(self):
        import numpy as np
        from horns import correlation_matrices
        from scipy.linalg import issymmetric

        a = np.random.rand(100, 100, 10)
        out = correlation_matrices(a)
        self.assertEqual(out.shape, (100, 10, 10))
        for i in range(100):
            self.assertTrue(issymmetric(out[i]))
        a = np.digitize(a, bins=np.quantile(a, [0.1, 0.5, 0.9]))
        out = correlation_matrices(a)
        self.assertEqual(out.shape, (100, 10, 10))
        for i in range(100):
            self.assertTrue(issymmetric(out[i]))

    def test_eigenvalues(self):
        import numpy as np
        from horns import eigenvalues, pearson_matrix

        a = np.random.rand(1000, 10)
        a = pearson_matrix(a)
        out = eigenvalues(a, "pca")
        out2 = eigenvalues(a, "fa")
        self.assertTrue(np.all(out > out2))

    def test_eigenvalues_all(self):
        import numpy as np
        from horns import all_eigenvalues, pearson_matrices

        a = np.random.rand(10, 100, 10)
        a = pearson_matrices(a)
        out = all_eigenvalues(a, "pca")
        out2 = all_eigenvalues(a, "fa")
        self.assertTrue(np.all(out > out2))

    def test_parallel_analysis(self):
        import numpy as np
        from horns import parallel_analysis
        import logging  # noqa

        a = np.random.rand(1000, 10)
        a[:, 0] = np.digitize(
            a[:, 0], bins=np.quantile(a[:, 0], [0.1, 0.5, 0.9])
        ).astype(float)
        # logging.basicConfig(level=logging.DEBUG)
        out = parallel_analysis(a, simulations=int(1e3), full_output=True)
        self.assertIsInstance(out, dict)
        # out["figure"].show()
