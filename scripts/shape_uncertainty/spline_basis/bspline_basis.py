from scipy import interpolate, special
import numpy as np


def make_knots(n_interior, T, degree=3):
    """Generate the augmented knot sequence tau for a B-spline basis on [0, T].

    The Cox-de Boor recursion is defined on an augmented knot sequence
    consisting of K interior knots, here spaced uniformly on the open
    interval (0, T), plus degree+1 repeated copies of each boundary
    knot (0 and T). The repetition at the boundaries makes the basis
    span the full spline space on [0, T], so that the resulting basis
    has dimension B = K + degree + 1.

    Args:
        n_interior: int, number of interior knots K.
        T: float, right end of the time domain [0, T].
        degree: int, spline degree (default 3, i.e. cubic).

    Returns:
        tau: np.ndarray of shape (K + 2 * (degree + 1),), the
            non-decreasing augmented knot sequence
            [0, ..., 0, xi_1, ..., xi_K, T, ..., T].
    """
    interior = np.linspace(0, T, n_interior+2)[1:-1]
    tau = np.concatenate([
        np.zeros(degree + 1),
        interior,
        np.full(degree + 1, T),
    ])
    assert list(tau.shape)[0] == n_interior + 2*(degree+1)
    return tau


def basis_functions(knots, degree=3):
    """Construct the individual B-spline basis functions phi_1, ..., phi_B.

    We construct B-spline basis functions using the BSpline function from scipy.interpolate. 
    However, the BSpline object represents a full spline, i.e. a weighted sum
    w_1 phi_1(t) + ... + w_B phi_B(t), specified by a knot sequence and
    a coefficient vector w. We obtain the single basis function phi_b
    by choosing the coefficient vector e_b = (0, ..., 0, 1, 0, ..., 0)
    with the 1 in position k. We return a list of B.spline objects corresponding
    to B-spline basis functions.

    Args:
        knots: np.ndarray, augmented knot sequence from make_knots.
        degree: int, spline degree (default 3).

    Returns:
        list of B scipy.interpolate.BSpline objects,
        B = len(knots) - degree - 1. = K + degree + 1, where K = nr. interior knots.
    """
    # Create the B x B identity matrix with rows being the coefficent vectors e_b.
    B = len(knots) - degree - 1
    coefficient_matrix = np.eye(B)
    # Create a list of Bspline basis function objects. 
    bspline_functions = [interpolate.BSpline(knots, coefficient_matrix[b], degree, extrapolate=False) for b in range(0, B)]
    assert len(bspline_functions) == B
    return bspline_functions


def basis_matrix(t, basis_functions):
    """Generate the design matrix Phi_i. 

    The model is y_i = Phi_i (h_theta(x_i) + u_i) + eps_i.
    The design matrix Phi_i is of dimension N_i x B. Each row j
    consists of a row vector of the B-spline basis functions evaluated
    at observation time t_j for individual i. 

    Args:
        t: np.ndarray of shape (N,), evaluation times in [0, T].
        basis_functions: list of B-spline basis function (BSpline objects)

    Returns:
        Phi: np.ndarray of shape (N, B) with B = len(basis_functions),
            where Phi[j, k] = phi_k(t_j).
    """
    t = np.asarray(t)
    basis_matrix_phi = np.column_stack([phi(t) for phi in basis_functions])
    assert basis_matrix_phi.shape == (len(t), len(basis_functions))
    return basis_matrix_phi


def derivative_basis_functions(basis_functions, nu):
    """Differentiate each basis function nu times.

    Args:
        basis_functions: list of B BSpline objects.
        nu: int, derivative order (nu=1 slope, nu=2 curvature).

    Returns:
        list of B BSpline objects, the nu-th derivatives.
    """
    return [phi.derivative(nu) for phi in basis_functions]


def basis_matrix_d2(t, basis_functions):
    """Generate the second-derivative design matrix Phi''.

    The analogue of basis_matrix for curvature: each row j is the row
    vector of second derivatives of the B basis functions evaluated at
    time t_j, i.e. Phi''[j, b] = phi_b''(t_j). 

    Note: rows of Phi'' sum to 0 (not 1), since the basis functions sum
    to the constant function 1, whose second derivative is 0. 

    Args:
        t: np.ndarray of shape (N,), evaluation times in [0, T].
        basis_functions: list of B BSpline objects from basis_functions().

    Returns:
        Phi_dd: np.ndarray of shape (N, B), B = len(basis_functions),
            where Phi_dd[j, b] = phi_b''(t_j).
    """
    return basis_matrix(t, derivative_basis_functions(basis_functions, 2))


def penalty_matrix(basis_functions, knots):
    """Return the smoothness penalty matrix Omega.

    Omega is defined entrywise as
        Omega[a, b] = \\int_0^T phi_a''(t) phi_b''(t) dt,
    so that for a coefficient vector w, the quadratic form w^T Omega w
    equals the integrated squared curvature \\int_0^T (y''(t))^2 dt of
    the spline y(t) = w^T phi(t). It enters the training objective via
    the mean-roughness term h_theta^T Omega h_theta and the
    random-effects term tr(Omega Sigma).

    Scipy cannot multiply two BSpline objects to form the integrand 
    as a new spline, so instead of an antiderivative we use exact 
    Gauss-Legendre quadrature. For a cubic spline each
    phi_b'' is linear on every knot interval, hence each product
    phi_a'' phi_b'' is quadratic on every knot interval. A 2-point
    Gauss-Legendre rule integrates polynomials up to degree 3 with zero
    error, so applying it per knot interval and summing over the K+1
    intervals yields Omega. The nodes must be placed per interval: 
    the integrand is only piecewise quadratic, so
    a single global rule on [0, T] would be inexact.

    Properties (used as tests): Omega is symmetric positive
    semi-definite with a 2-dimensional null space, spanned by the
    coefficient vectors representing the functions 1 and t (zero
    curvature everywhere).

    Args:
        basis_functions: list of B BSpline objects from basis_functions().
        knots: np.ndarray, augmented knot sequence from make_knots
            (needed to locate the knot intervals for the per-interval
            quadrature).

    Returns:
        Omega: np.ndarray of shape (B, B), symmetric PSD,
            B = len(basis_functions).
    """
    # Standard Gauss-Lagrange nodes and weights
    gl_nodes = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gl_weights = np.array([1, 1])

    # We generate an array of shifted nodes and weights, respectively for all intervals.
    # This is because we compute the integral over [0, T] rather than standard GL [-1, 1].
    stripped_knots = np.unique(knots)
    knot_midpoints = [(a + b) / 2 for a, b in zip(stripped_knots, stripped_knots[1:])]
    knot_widths = [(b - a) for a, b in zip(stripped_knots, stripped_knots[1:])]

    node1 = [midpoint + (width/2) * gl_nodes[0] for midpoint, width in zip(knot_midpoints, knot_widths)]
    node2 = [midpoint + (width/2) * gl_nodes[1] for midpoint, width in zip(knot_midpoints, knot_widths)]
    shifted_nodes = np.array(list(zip(node1, node2)))

    weight1 = [(width/2) * gl_weights[0]  for width in knot_widths]
    weight2 = [(width/2) * gl_weights[1]  for width in knot_widths]
    shifted_weights = np.array(list(zip(weight1, weight2)))

    # Initialise omega as a zero matrix of BxB dimension:
    B = len(basis_functions)
    omega = np.zeros((B, B))

    # Compute second order derivatives of splines:
    phi_double_prime = derivative_basis_functions(basis_functions, 2)

    # Evaluate integral interval-by-interval:
    for i in range(0, len(knot_midpoints)):  
        # Evaluate 2nd derivatives of all B basis functions at 2 nodes in inteval i:
        nodes_i = np.asarray(shifted_nodes[i]) # (2, )
        phi_eval = basis_matrix(nodes_i, phi_double_prime) # (2, B)

        # Compute outer products for node 1 and node 2
        f_node1 = np.outer(phi_eval[0], phi_eval[0])  # (B, B)
        f_node2 = np.outer(phi_eval[1], phi_eval[1])  # (B, B)
    
        # Apply weights and accumulate into the global omega matrix
        weights_i = np.asarray(shifted_weights[i]) # (2, )
        omega += weights_i[0] * f_node1 + weights_i[1] * f_node2

    return omega


def monomial_coefficents(basis_functions, knots):
    """Precompute per-piece monomial conversion matrices for shape summary extraction.

    A cubic spline is piecewise cubic. That means, on each knot interval
    [xi_k, xi_{k+1}), k = 0, ..., K, the trajectory y(t) = w^T phi(t)
    restricts to a single cubic polynomial:
                \\hat{y} = c_0 + c_1 t + c_2 t^2 + c_3 t^3
    Shape summary extraction needs this polynomial form.

    For each piece k this function returns a matrix C_k of shape
    (4, B) such that, 

        y(t) = sum_{j=0}^{3} (C_k @ w)_j * t**j,   t in [xi_k, xi_{k+1}).

    Procedure:
        The conversion is computed exactly in two stages. First, Stage 1 uses 
        Taylor's Theorem to extract exact local coefficients centered at the left 
        edge of each interval (xi_k) by evaluating the normalised derivatives 
        (0 through 3) of the B-spline basis functions. Because the basis functions 
        are pure cubic polynomials within each interval, this Taylor expansion has 
        zero analytical error. Second, Stage 2 applies a hardcoded 4x4 binomial 
        expansion matrix to shift these local coordinates (t - xi_k)^j into global 
        monomial powers of t^j.

    Args:
        basis_functions: list of B BSpline objects from basis_functions().
        knots: np.ndarray, augmented knot sequence from make_knots.

    Returns:
        C: np.ndarray of shape (K + 1, 4, B), where C[k] is the
            conversion matrix for the k-th knot interval, ordered left
            to right over the K + 1 intervals of [0, T].

    Note: The monomial coefficients are ordered from lowest to highest
        [c0, c1, c2, c3]
    """
    # --------------------------------------------------------------------------
    # Stage 1: Taylor Expansion - Exact as the functions are polynomals
    # \phi_b(t)= \phi_b(\xi_k)/0! + \phi_b^'(\xi_k)/1! * (t-xi_k) 
    #        + \phi_b^''(\xi_k)/2! * (t-xi_k)^2 + \phi_b^'''(\xi_k)/3! * (t-xi_k)^3
    # --------------------------------------------------------------------------
    
    # We initialise a tensor of LOCAL polynomial coefficents we set out to fill.
    # The polynomial coefficents are relative to the local point of the approximation
    intervals = len(np.unique(knots))-1 
    B = len(basis_functions)
    C_local = np.zeros((intervals, 4, B)) 

    # We create a list of evaluation points for the Taylor approximations. 
    # Splines are defined on [xi_i, xi_{i+1}) so use the left boundary of the intervals.
    left_edges = np.unique(knots)[:-1] 

    # Loop over the 4 polynomial degrees:
    for j in range(0, 4):
        # Compute jth order derivative
        phi_prime_j = derivative_basis_functions(basis_functions, j)
        # Evaluate derivative on the left edge
        derivative_evaluation = basis_matrix(left_edges, phi_prime_j)
        # Replace the affected entires in with the coefficents of the Taylor approximation
        C_local[:, j, :] = derivative_evaluation / special.factorial(j)
        
    # --------------------------------------------------------------------------
    # Stage 2: Multiplying out the Taylor approximation using a shift matrix
    # --------------------------------------------------------------------------
    # We initialise a tensor of GLOBAL polynomial coefficents that we set out to fill.
    C_global = np.zeros_like(C_local)

    for k in range(0, intervals):
        # We define a upper-triangular binomial shift matrix S(xi). Multiplying by this
        # is analogous to multiplying out the polynomials.
        xi = left_edges[k]
        S = np.array([
            [1.0, -xi,   xi**2,  -xi**3],
            [0.0,  1.0, -2.0*xi,  3.0*(xi**2)],
            [0.0,  0.0,   1.0,   -3.0*xi],
            [0.0,  0.0,   0.0,    1.0]
        ])

        # S is (4, 4) and C_local[k] is (4, B), yielding a (4, B) matrix output
        C_global[k] = S @ C_local[k]

    print(C_global)
    return C_global



if __name__ == "__main__":
    knots = make_knots(2, 5, degree=3)
    bsplines = basis_functions(knots, degree=3)
    penalty_matrix(bsplines, knots)
    monomial_coefficents(bsplines, knots)