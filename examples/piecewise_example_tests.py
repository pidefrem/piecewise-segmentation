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


def check_square_error(weights, values, reg, error, epsilon=1e-14):
    """
    Recompute square error
    """
    check_square_error = sum(weights * (values - reg)**2)
    print("[INFO] Square error: {0}".format(error))
    print('[DEBUG] Check Square error for DEBUG: {0}'.format(check_square_error))

    if abs(check_square_error - error) >= epsilon:
        print('[DEBUG] Recomputed square error is different from algorithm square error')


def check_bias(weights, values, reg, clusters, epsilon=1e-14):
    """
    Check for bias in the clusters
    """
    clust_init_means = {}
    clust_means = {}
    clust_weights = {}

    for index, clust in enumerate(list(clusters)):
        if clust not in clust_init_means.keys():
            clust_init_means[clust] = 0
            clust_means[clust] = 0
            clust_weights[clust] = 0
        clust_init_means[clust] += weights[index] * values[index]
        clust_means[clust] += weights[index] * reg[index]
        clust_weights[clust] += weights[index]

    nb_init_clust_error = 0
    for clust in clust_init_means.keys():
        if abs(clust_init_means[clust] - clust_means[clust]) >= epsilon:
            nb_init_clust_error += 1

    if nb_init_clust_error > 0:
        print('[DEBUG] nb_init_clust with bias error: {0}'.format(nb_init_clust_error))
        print("clust_init_means: {0}".format(clust_init_means))
        print("clust_means: {0}".format(clust_means))


def check_reg(reg, expected_reg, epsilon=1e-6):
    N = reg.shape[0]
    if expected_reg.shape[0] != N:
        raise Exception("check_reg: series must have same length")
    for i in range(N):
        if abs(reg[i] - expected_reg[i]) >= epsilon:
            print("[DEBUG] reg and expected_reg differ")
            break


def test_regress_ksegments(k, series, weights=None, expected_regs=None, all_clusters=False, plot_reg=True, method='twopass'):
    print("[INFO] test_regress_ksegments")

    reg_list, clusters_list, error_list = regress_ksegments(k, values, weights, all_clusters, method)

    print("[INFO] Output:")
    for k_prime in range(k):
        reg, clusters, error = reg_list[k_prime], clusters_list[k_prime], error_list[k_prime]

        if expected_regs and (k - k_prime) not in expected_regs:
            raise Exception("Missing key in expected_regs: ", k-k_prime)
        expected_reg = expected_regs[k - k_prime] if expected_regs else None

        print("[INFO] k: {0}; Reg: {1}; size: {2}".format(k - k_prime, reg, reg.shape))
        # print("Clusters: {0}; size: {1}".format(clusters, clusters.shape))

        check_square_error(weights, series, reg, error)

        check_bias(weights, series, reg, clusters)

        if expected_reg is not None:
            check_reg(reg, expected_reg)

        if plot_reg:
            plot_data(weights, series, reg, k, k_prime)

        if not all_clusters:
            break

    return


if __name__ == "__main__":
    print("Bellman K-segments")

    # default linewidth=75
    np.set_printoptions(precision=10, suppress=True, linewidth=100)

    # Test 1 (sinusoidal function)
    print('[INFO] Test 1 (sinusoidal function)')
    N = 14
    x = np.arange(0, N)
    values = np.sin(x * 2 * np.pi / (N-1)) + 1
    weights = np.ones(values.shape, dtype=np.float) / np.size(values)

    k = 6
    expected_regs = {
        6: np.array([1.232361586, 1.232361586, 1.8534579102, 1.8534579102, 1.8534579102, 1.8534579102, 1.2393156643, 0.7606843357, 0.1465420898, 0.1465420898, 0.1465420898, 0.1465420898, 0.767638414, 0.767638414 ]),
        5: np.array([1.232361586, 1.232361586, 1.8534579102, 1.8534579102, 1.8534579102, 1.8534579102, 1., 1., 0.1465420898, 0.1465420898, 0.1465420898, 0.1465420898, 0.767638414, 0.767638414 ]),
        4: np.array([1., 1.6863117462, 1.6863117462, 1.6863117462, 1.6863117462, 1.6863117462, 1.6863117462, 0.3136882538, 0.3136882538, 0.3136882538, 0.3136882538, 0.3136882538, 0.3136882538, 1.          ]),
        3: np.array([1.588267211, 1.588267211, 1.588267211, 1.588267211, 1.588267211, 1.588267211, 1.588267211, 0.3136882538, 0.3136882538, 0.3136882538, 0.3136882538, 0.3136882538, 0.3136882538, 1.          ]),
        2: np.array([1.588267211, 1.588267211, 1.588267211, 1.588267211, 1.588267211, 1.588267211, 1.588267211, 0.411732789, 0.411732789, 0.411732789, 0.411732789, 0.411732789, 0.411732789, 0.411732789]),
        1: np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    }
    test_regress_ksegments(k, values, weights, expected_regs=expected_regs, all_clusters=True)

    print()

    # Test 2 (null weights, numerical instability (negative square error))
    print('[INFO] Test 2 (null weights, numerical instability (negative square error)')
    values = np.array([0,2.523170478,2.523170478,0,0,1.758323491,2.275910579,2.275910579,
                       2.523170478,2.523170478,2.523170478,2.523170478,0.908668713,0.908668713,
                       0.908668713,0.908668713,2.523170478,
                       2.523170478,2.523170478,2.523170478,2.523170478,2.523170478,2.523170478,
                       2.523170478,2.523170478])
    weights = np.array([0,1,1,0,0,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    k = 8
    expected_regs = {
        8: np.array([0., 2.523170478, 2.523170478, 1.758323491, 1.758323491, 1.758323491, 2.275910579, 2.275910579, 2.523170478, 2.523170478, 2.523170478, 2.523170478, 0.908668713, 0.908668713, 0.908668713, 0.908668713, 2.523170478, 2.523170478, 2.523170478, 2.523170478, 2.523170478, 2.523170478, 2.523170478, 2.523170478, 2.523170478])
    }
    try:
        test_regress_ksegments(k, values, weights, expected_regs=expected_regs, all_clusters=False)
    except BaseException as e:
        print("[ERROR] Test 2: ", e)
        plot_input(values, weights)
    print()

    # Test 3 (null weights, k=8: less points than clusters, numerical instability (negative square error))
    print('[INFO] Test 3 (null weights, k=8: less points than clusters, numerical instability (negative square error))')
    values = np.array([0.900068122,0,0.900068122,0.473194682,0.046321243,0.046321243,0.046321243,0,0,0.056904085,
                       0.900068122,0.900068122,0.900068122,0.900068122,0,0,0,0,0,
                       0,0,0,0,0,0,0.900068122,0.900068122,0.900068122,0.900068122,
                       0.900068122,0.900068122,0.900068122,0.900068122])
    weights = np.array([1,0,1,2,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,
                        0,0,0,0,1,1,1,1,1,1,1,1])
    k = 8
    expected_regs = {
        8: np.array([0.900068122, 0.900068122, 0.900068122, 0.473194682, 0.046321243, 0.046321243, 0.046321243, 0.056904085, 0.056904085, 0.056904085, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122]),
        7: np.array([0.900068122, 0.900068122, 0.900068122, 0.473194682, 0.046321243, 0.046321243, 0.046321243, 0.056904085, 0.056904085, 0.056904085, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122]),
        6: np.array([0.900068122, 0.900068122, 0.900068122, 0.473194682, 0.046321243, 0.046321243, 0.046321243, 0.056904085, 0.056904085, 0.056904085, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122]),
        5: np.array([0.900068122, 0.900068122, 0.900068122, 0.473194682, 0.046321243, 0.046321243, 0.046321243, 0.056904085, 0.056904085, 0.056904085, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122]),
        4: np.array([0.900068122, 0.900068122, 0.900068122, 0.473194682, 0.0489669535, 0.0489669535, 0.0489669535, 0.0489669535, 0.0489669535, 0.0489669535, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122 ]),
        3: np.array([0.686631402, 0.686631402, 0.686631402, 0.686631402, 0.0489669535, 0.0489669535, 0.0489669535, 0.0489669535, 0.0489669535, 0.0489669535, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122 ]),
        2: np.array([0.3677991777, 0.3677991777, 0.3677991777, 0.3677991777, 0.3677991777, 0.3677991777, 0.3677991777, 0.3677991777, 0.3677991777, 0.3677991777, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122 ]),
        1: np.array([0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443, 0.6871605443])
    }
    try:
        test_regress_ksegments(k, values, weights, expected_regs=expected_regs, all_clusters=True)
    except BaseException as e:
        print("[ERROR] Test 3: ", e)
        plot_input(values, weights)
    print()

    # Test 4
    print('[INFO] Test 4')
    values = np.array([0.900068122,0.473194682,0.046321243,0.046321243,0.046321243,0,0,0.056904085,
                       0.900068122,0.900068122,0,0,
                       0.900068122,0.900068122,0])
    weights = np.array([1,2,1,1,1,0,0,1,1,1,0,
                        0,1,1,0])
    k = 8
    expected_regs = {
        8: np.array([0.900068122, 0.473194682, 0.046321243, 0.046321243, 0.046321243, 0.056904085, 0.056904085, 0.056904085, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122])
    }
    try:
        test_regress_ksegments(k, values, weights, expected_regs=expected_regs, all_clusters=False)
    except BaseException as e:
        print("[ERROR] Test 4: ", e)
        plot_input(values, weights)
    print()

    # Test 5 (k=7: less points than clusters)
    print('[INFO] Test 5 (k=7: less points than clusters)')
    values = np.array([0.900068122,0.473194682,0.046321243,0,0.096904085,
                       0.900068122,0,0,
                       0.900068122,0])
    weights = np.array([1,2,1,0,1,1,0,0,1,0])

    k = 7
    expected_regs = {
        7: np.array([0.900068122, 0.473194682, 0.046321243, 0., 0.096904085, 0.900068122, 0.900068122, 0.900068122, 0.900068122, 0.900068122])
    }
    try:
        test_regress_ksegments(k, values, weights, expected_regs=expected_regs, all_clusters=False)
    except BaseException as e:
        print("[ERROR] Test 5: ", e)
        plot_input(values, weights)

    print()

    # Test 6 (valley and plateau with null weights)
    print('[INFO] Test 6 (valley and plateau with null weights)')
    values = np.array([9.00068122,4.73194682,0.46321243,0,0.96904085,
                       9.00068122,0,0,
                       9.00068122,0])
    weights = np.array([1,2,1,0,1,1,0,0,1,0])

    k = 6
    expected_regs = {
        6: np.array([9.00068122, 4.73194682, 0.46321243, 0., 0.96904085, 9.00068122, 9.00068122, 9.00068122, 9.00068122, 9.00068122])
    }
    try:
        test_regress_ksegments(k, values, weights, expected_regs=expected_regs, all_clusters=False)
    except BaseException as e:
        print("[ERROR] Test 6: ", e)
        plot_input(values, weights)

    print()
