# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
"""
Examples using piecewisesegmentation
"""

import numpy as np

from piecewisesegmentation.piecewisesegmentation import regress_ksegments

import matplotlib.pyplot as plt


def plot_input(values, weights):
    """
    Line plot of the input data
    """
    N = values.shape[0]

    # Plot data (for k clusters):
    plt.style.use('ggplot')
    # with plt.style.context(''):
    plt.plot(values, 'ro')
    plt.plot(weights, 'bx')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Input values and weights')
    axes = plt.gca()
    xmin, xmax = -1, N
    ymin, ymax = min(min(values), min(weights)) - 0.2, max(max(values), max(weights)) + 0.2
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.show()

    return


def plot_data(weights, values, reg, k, k_prime=0):
    """
    Line plot of the regression result
    """
    N = values.shape[0]
    # Plot data (for k clusters):
    plt.style.use('ggplot')
    # with plt.style.context(''):
    plt.plot(values, 'ro')
    plt.plot(weights, 'bx')
    plt.plot(reg)
    # plt.plot(correct_reg, color='b')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Clustering for k={0}'.format(k - k_prime))
    axes = plt.gca()
    xmin, xmax = -1, N
    ymin, ymax = min(min(values), min(weights)) - 0.2, max(max(values), max(weights)) + 0.2
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.show()


if __name__ == "__main__":
    print("Bellman K-segments")

    np.set_printoptions(precision=10, suppress=True, linewidth=100)

    # Test 1 (sinusoidal function)
    print('[INFO] Test 1 (sinusoidal function)')
    N = 14
    x = np.arange(0, N)
    values = np.sin(x * 2 * np.pi / (N-1)) + 1
    weights = np.ones(values.shape, dtype=np.float) / np.size(values)
    plot_input(values, weights)

    k = 6

    # Simple example: get the regression for k segments
    print("[INFO] Simple example")
    reg_list, clusters_list, error_list = regress_ksegments(k, values, weights)
    reg, clusters, error = reg_list[0], clusters_list[0], error_list[0]

    print("[INFO] k: {0}; Reg: {1}; size: {2}".format(k , reg, reg.shape))
    print("Clusters: {0}; size: {1}".format(clusters, clusters.shape))

    plot_data(weights, values, reg, k, 0)

    # More advanced: get all regressions for k_prime<=k
    print("[INFO] More advanced example")
    reg_list, clusters_list, error_list = regress_ksegments(k, values, weights, all_clusters=True)

    for k_prime in range(k):
        reg, clusters, error = reg_list[k_prime], clusters_list[k_prime], error_list[k_prime]

        print("[INFO] k: {0}; Reg: {1}; size: {2}".format(k - k_prime, reg, reg.shape))
        print("Clusters: {0}; size: {1}".format(clusters, clusters.shape))

        plot_data(weights, values, reg, k, k_prime)
