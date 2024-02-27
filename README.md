# Horns: Horn's parallel analysis in Python

Horns is a Python implementation of Horn's (1965) parallel analysis, the most widely
accepted method for determining the number of components or factors to retain in
principal component analysis (PCA) or common factor analysis (FA). The functionality of
this package is similar to that of the `paran` package in R.

## Background

Parallel analysis involves simulating a large number of random datasets with the same 
shape as the original dataset but with no underlying correlation structure. We calculate
the eigenvalues of the random datasets and the *q*th quantile of the distribution of 
each eigenvalue, as well as the eigenvalues of the original dataset. The original 
eigenvalues are then compared to the quantiles. The number of components/factors to
retain is the number of original eigenvalues that are greater than their corresponding
quantile until we encounter the first eigenvalue that is not greater than its quantile.

Horn (1965) originally proposed using the median at the selection criterion (i.e.,
$q=0.5$), but Glorfeld (1995) recommended *q* = 0.95 (and a large number of 
simulations) to reduce the chances of retaining too many components or factors. As in
`paran`, the user can choose *q* and the number of simulations, allowing them to follow
Glorfeld's recommendations or not.

There has been some debate about the best way to simulate random data for parallel 
analysis. Hayton et al.(2004) originally claimed it is necessary to simulate data with 
the same values of the original data, but later Dinno (2009) demonstrated that 
parallel analysis is robust to a wide range of distributional forms of the random data,
and therefore recommended using the most computationally efficient method available.
This may be good advice when one is performing parallel analysis on Pearson correlation
or covariance matrices, but I'm not sure it makes sense for other kinds of matrices
(i.e., polyserial or polychoric correlations). Therefore, I have included several 
methods of simulating random data, including shuffling and bootstrapping the original
data.

PCA and FA, and therefore parallel analysis, are often performed on Pearson correlation
matrices. However, a Pearson correlation matrix is not the correct choice in all cases. 
For example, Tran et al. (2008) showed that parallel analysis on binary data is more
accurate when using polychoric correlation matrices. This package will select the
appropriate correlation estimate per pair of variables based on the number of unique
values a each variable.

As pointed out by Dinno (2014), some implementations of parallel analysis do not
correctly calculate the eigenvalues for FA, which are different from those for PCA. This
package uses the correct eigenvalues for both PCA and FA, like the `paran` package in R.

Horns optionally produces a figure showing the eigenvalues and the quantiles via 
Matplotlib. 

### Performance

Since there are apparently no other Python packages that perform parallel analysis, I 
didn't profile or benchmark my code extensively. However, the package does perform 
just-in-time (JIT) compilation of many of its functions via Numba, and parallelises
where possible, so it should be reasonably fast. Parallel analysis with polychoric 
correlations does take much longer than with Pearson and/or polyserial correlations
because each correlation is found iteratively. 

## Installation

You can install Horns directly from PyPI using pip:

```bash
pip install horns
```

## Quick Start

Here's a quick example to get you started:

```python

import pandas as pd  # <- not required by Horns, but you need to load your data somehow
from horns import parallel_analysis

# load your dataset
data = pd.read_csv("path/to/your/data.csv")

# perform parallel analysis to determine the optimal number of components for PCA
m = parallel_analysis(data)

print(f"Optimal number of components: {m}")

```

There should be no need to call anything other than `parallel_analysis`, but you may 
find some of the ancillary functions useful for other applications.

## Contributing

Contributions to Horns are welcome! Submit an issue or pull request if you have any
suggestions or would like to contribute.

## License

This project is licensed under the MIT License.

## Citation

If you use Horns in your research, please consider citing it:

```bibtex

@misc{horns2024,
  title={Horns: Horn's parallel analysis in Python},
  author={Samuel R. Mathias},
  year={2024},
  howpublished={\url{https://github.com/sammosummo/Horns}},
}
```

## Thanks

Thanks for choosing Horns for your factor analysis needs!