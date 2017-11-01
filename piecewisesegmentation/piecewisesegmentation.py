# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
"""
Bellman, k-segmentation algorithm

Based on initial work in Julia by Justin Willmert:
http://homepages.spa.umn.edu/~willmert/science/ksegments/

References:
"Niina Haiminen, Aristides Gionis, & Kari Laasonen. Algorithms for unimodal segmentation with applications to unimodality detection.
Knowledge and Information Systems, 2008, Vol.14(1), pp.39-57 <http://link.springer.com/article/10.1007/s10115-006-0053-3>)"
"Richard Bellman. On the approximation of curves by line segments using dynamic programming.
Communications of the ACM, 4(6), 1961."

Objective:
Find the sequence segmentation that minimizes:
.. math::
    \begin{align}
        E_S = \sum_{i=1}^N w_i (x_i - \bar x_i)^2
    \end{align}

Principle:
Using dynamic programming.

Algorithm details:
Let $E_S[i,k']$ represent the segmentation error over the data points $\{x_1,\ldots,x_i\}$ using $k'$ segments,
and let $E[i,j]$ be the error in representing the points $\{x_i,\ldots,x_j\}$ using just the mean of the data.

For the trivial case of $k' = 1$, we simply fit a single line segment as the mean over all data points,
with an associated error $E_S[N,1] = E[1,N]$. 

Next, allow a second line segment to be used.
The question becomes at which point i do we add the line segment so that there are two line segments from $1...i$ and $(i+1)...N$.
The error in this case is then
$$\begin{align*}
    E_S[N,2] = \min_{1\le i\le N} ( E[1,i] + E[i+1,N] )
\end{align*}$$

Moving up to $k' = 3$, let’s recast the previous two cases in terms of “growing” sequences:
$$\begin{align*}
    E_S[i,1] &= E[1,i] \\
    E_S[i,2] &= \min_{1\le j \le i} ( E_S[j-1,1] + E[j,i] ) \\
\end{align*}$$

Through induction, we obtain the generic recursion relation.
$$\begin{align}
    E_S[i,k'] &= \min_{1\le j \le i} ( E_S[j-1,k'-1] + E[j,i] )
        \label{eqn:recurse}
\end{align}$$
"""

import numpy as np


def _compute_subseq_errors_direct(series, weights):
    """
    Subsequence errors (using one pass formulation)

    :param Array{Float64,1} series
    :param Array{Float64,1} weights

   The subsequence errors is:
    $$\begin{align}
        E[i,j] &= Q[i,j] - \frac{S[i,j]^2}{W[i,j]}
    \end{align}$$

    Were W, S, Q are upper diagonal matrices:
    $$\begin{align}
        W[i,j] &\equiv \sum_{k=i}^j w_k \\
        S[i,j] &\equiv \sum_{k=i}^j w_k x_k \\
        Q[i,j] &\equiv \sum_{k=i}^j w_k {x_k}^2
    \end{align}$$

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    Because $$Q[i,j]$$ and $$\frac{S[i,j]^2}{W[i,j]}$$ can be very similar numbers,
    cancellation can lead to the precision of the result to be much less than
    the inherent precision of the floating-point arithmetic used to perform the computation.
    Thus this algorithm should not be used in practice.
    This is particularly bad if the standard deviation is small relative to the mean.
    """
    N = np.size(series)

    wgts = np.diag(weights)
    wsum = np.diag(weights * series)
    sqrs = np.diag(weights * series * series)

    dists = np.zeros((N, N), dtype=np.float)
    means = np.diag(series)

    # Fill the upper triangle of dists and means by performing up-right
    # diagonal sweeps through the matrices
    for delta in range(0, N):
        for l in range(0, (N-1-delta)):
            # l = left boundary, r = right boundary
            r = l + delta + 1

            # Incrementally update every partial sum
            wgts[l, r] = wgts[l, r-1] + wgts[r, r]
            wsum[l, r] = wsum[l, r-1] + wsum[r, r]
            sqrs[l, r] = sqrs[l, r-1] + sqrs[r, r]

            # Calculate the mean over the range
            means[l, r] = 0 if (wgts[l, r] == 0) else wsum[l, r] / wgts[l, r]
            dists[l, r] = sqrs[l, r] - means[l, r] * wsum[l, r]

            if dists[l, r] < 0:
                print("[WARNING] Numerical instability detected, dists[", l, ", ", r, "] is negative: ", dists[l, r])

    return dists, means


def _compute_subseq_errors_wessel(series, weights):
    """
    Subsequence errors (using Wessel formula)

    :param Array{Float64,1} series
    :param Array{Float64,1} weights

    We need to compute the three following upper diagonal matrices:
    $$\begin{align}
        W[i,j] &\equiv \sum_{k=i}^j w_k \\
        S[i,j] &\equiv \sum_{k=i}^j w_k x_k \\
        M[i,j] &\equiv \frac{\sum_{k=i}^j w_k x_k}{\sum_{k=i}^j w_k}  \\
    \end{align}$$

    In order to avoid catastrophic cancellation (numerical instability),
    we use Wessel formula for the variance.
    Then the subsequence errors:
    $$\begin{align}
        E[i,j] &= E[i,j-1] + w_j(x_j - M[i,j-1])(x_j - M[i,j])
    \end{align}$$
    """
    N = np.size(series)

    wgts = np.diag(weights)
    wsum = np.diag(weights * series)
    sqrs = np.diag(weights * series * series)

    dists = np.zeros((N, N), dtype=np.float)
    means = np.diag(series)

    # Fill the upper triangle of dists and means by performing up-right
    # diagonal sweeps through the matrices
    for delta in range(0, N):
        for l in range(0, (N-1-delta)):
            # l = left boundary, r = right boundary
            r = l + delta + 1

            # Incrementally update every partial sum
            wgts[l, r] = wgts[l, r-1] + wgts[r, r]
            wsum[l, r] = wsum[l, r-1] + wsum[r, r]
            sqrs[l, r] = sqrs[l, r-1] + sqrs[r, r]

            # Calculate the mean over the range
            means[l,r] = 0 if (wgts[l, r] == 0) else means[l, r-1] + (wgts[r, r] / wgts[l, r]) * (series[r] - means[l, r-1])
            # Then update the distance calculation.
            dists[l, r] = dists[l, r-1] + wgts[r, r] * (series[r] - means[l, r - 1]) * (series[r] - means[l, r])

            if dists[l, r] < 0:
                print("[WARNING] Numerical instability detected, dists[", l, ", ", r, "] is negative: ", dists[l, r])

    return dists, means


def _compute_subseq_errors_twopass(values, weights):
    """
    Subsequence errors (two-pass computation)

    :param Array{Float64,1} values
    :param Array{Float64,1} weights
    """
    def _weighted_avg_and_distance(values, weights):
        if weights.sum() == 0:
            average = 0
            dist = 0
        else:
            average = np.average(values, weights=weights)
            dist = np.dot((values - average)**2, weights)
        return average, dist

    N = np.size(values)

    means = np.diag(values)
    dists = np.zeros((N, N), dtype=np.float)

    for i in range(0, N-1):
        for j in range(i+1, N):
            (means[i, j], dists[i, j]) = _weighted_avg_and_distance(values[i:j+1], weights[i:j+1])

    return dists, means


def regress_ksegments(k, series, weights=None, all_clusters=False, method='wessel'):
    """
    Segmentations errors

    :param int k : number of segments
    :param Array{Float64,1} series: series values to regress
    :param Array{Float64,1} weights: weights for the regression
    :param bool all_clusters: if True, returns the results for each $$k'<=k$$
    :param str method: method for computing subsequences errors
    """

    k = int(k)
    series = series.astype('float64')
    if weights is None:
        weights = np.ones(np.size(series), dtype=np.float) / np.size(series)
    else:
        weights = weights.astype('float64')

    # Make sure we have a row vector to work with
    if np.size(series) == 1:
        # Only a scalar value
        raise Exception("series must have length > 1")

    # Ensure series and weights have the same size
    if series.shape != weights.shape:
        raise Exception("series and weights must have the same shape")

    # Make sure the choice of k makes sense
    if k < 1 or k > np.size(series):
        raise Exception("k must be in the range 1 to length(series)")

    N = np.size(series)

    # Get pre-computed distances and means for single-segment spans over any
    # arbitrary subsequence series(i:j). The costs for these subsequences will
    # be used *many* times over, so a huge computational factor is saved by
    # just storing these ahead of time.
    if method not in ['direct', 'wessel', 'twopass']:
        raise Exception("method for subsequences computation must be 'direct', 'wessel' or 'twopass'")
    if method == 'direct':
        (one_seg_dist, one_seg_mean) = _compute_subseq_errors_direct(series, weights)
    elif method == 'wessel':
        (one_seg_dist, one_seg_mean) = _compute_subseq_errors_wessel(series, weights)
    else:
        (one_seg_dist, one_seg_mean) = _compute_subseq_errors_twopass(series, weights)

    # Keep a matrix of the total segmentation costs for any p-segmentation of
    # a subsequence series[1:n] where 1<=p<=k and 1<=n<=N. The extra column at
    # the beginning is an effective zero-th row which allows us to index to
    # the case that a (k-1)-segmentation is actually disfavored to the
    # whole-segment average.
    k_seg_dist = np.zeros((k, N+1), dtype=np.float)
    # Also store a pointer structure which will allow reconstruction of the
    # regression which matches. (Without this information, we'd only have the
    # cost of the regression.)
    k_seg_path = - np.ones((k, N), dtype=np.int)

    # Initialize the case k=1 directly from the pre-computed distances
    k_seg_dist[0, 1:] = one_seg_dist[0, :]  # Rem: k_seg_dist[0, 1] <=> (p=1, n=1^st elt)

    # The index into the matrix is the right (inclusive) boundary of a segment,
    # and the value it contains is the left (exclusive) boundary.
    # Any path with only a single segment has a left (non-inclusive) boundary
    # at the zeroth element (index '-1').
    # Then for p segments through p elements, the left boundary for the (p) right boundary
    # case must obviously be (p-1).
    for i in range(k):
        k_seg_path[i, i] = i - 1  # update (sub-)diagonal values by overriding the square matrix
    k_seg_path[0, :] = -1

    # values k_seg_path[p, n] for n < p are undefined

    # Now go through all remaining subcases 1 < nb_segments <= k
    # We start at nb_segments = 2
    for p in range(1, k):  # p = nb_segments - 1
        # Update the substructure as successively longer subsequences are
        # considered.
        for n in range(p+1, N):  # n = nb_points - 1
            # WARNING: if we recompute the case n==p here, in case of null weights,
            #          bestidx can be <= n-1 instead of equal to n).
            # Enumerate the choices and pick the best one. Encodes the recursion
            # for even the case where j=1 by adding an extra boundary column on the
            # left side of k_seg_dist. The j-1 indexing is then correct without
            # subtracting by one since the real values need a plus one correction.
            choices = k_seg_dist[p-1, 0:n+1] + one_seg_dist[0:n+1, n].T
            bestidx = np.argmin(choices)
            bestval = choices[bestidx]

            # bestidx == 0 if the whole segment is preferred

            # Store the sub-problem solution. For the path, store where the (p-1)
            # case's right boundary is located.
            k_seg_path[p, n] = bestidx - 1

            # Then remember to offset the distance information due to the boundary
            # (ghost) cells in the first column.
            k_seg_dist[p, n+1] = bestval

    # Eventual complete regression
    # Now use the solution information to reconstruct the optimal regression.
    # Fill in each segment reg(i:j) in pieces, starting from the end where the
    # solution is known.
    # We start with the right-hand side pointing to the N-th element for the solution with k segments.
    # The path matrix tells us where the left-hand side of the optimal segment is to be located,
    # so we fill in the regression values over that segment with the appropriate mean.
    # We then repeat for the k−1 remaining segments by moving up a row in the matrix,
    # and start with the right-hand side to the immediate left of the just-generated segment.
    reg_list = []
    clusters_list = []
    k_seg_dist_list = []
    for k_prime in range(k, 0, -1):
        # If all_clusters parameter is True, get the complete regression for all values of k_prime in {1,...,k}
        if k_prime < k and not all_clusters:
            break

        reg = np.zeros(series.shape, dtype=np.float)
        clusters = - np.ones(N, dtype=np.int)
        rhs = np.size(reg) - 1  # N-1
        for p in range(k_prime-1, -1, -1):
            # Get the corresponding previous boundary
            if rhs < p:
                print("[DEBUG] Null weights found. There are less points than clusters:", rhs, " points left for ", p, " clusters")
                # In case of null weights, we can find ourselves in a situation
                # where there are less points than clusters.
                # To avoid getting an uninitialized path, we skip the current iteration.
                continue
            lhs = k_seg_path[p, rhs]

            # The pair (lhs,rhs] is now a half-open interval, so set it appropriately
            reg[lhs+1:rhs+1] = one_seg_mean[lhs+1, rhs]
            clusters[lhs+1:rhs+1] = lhs * np.ones(rhs - (lhs+1) + 1, dtype=np.int)

            # Update the right edge pointer
            rhs = lhs

        reg_list.append(reg)
        clusters_list.append(clusters)
        k_seg_dist_list.append(k_seg_dist[k_prime-1, N])

    return reg_list, clusters_list, k_seg_dist_list
