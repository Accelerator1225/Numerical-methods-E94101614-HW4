import numpy as np
from scipy.integrate import quad, dblquad
from numpy.polynomial.legendre import leggauss

# ==============================================
# HW4 - Numerical Methods
# ==============================================

# -------------------------
# Q1. Composite Integration Methods
# -------------------------

def question1():
    def f1(x):
        return np.sin(4 * x) * np.exp(x)

    a = 1
    b = 2
    h = 0.1
    n = int((b - a) / h)

    x = np.linspace(a, b, n + 1)
    midpoints = (x[:-1] + x[1:]) / 2

    trapezoidal = (h / 2) * (f1(a) + 2 * np.sum(f1(x[1:-1])) + f1(b))
    if n % 2 != 0:
        n += 1
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
    simpson = (h / 3) * (f1(a) + 4 * np.sum(f1(x[1:-1:2])) + 2 * np.sum(f1(x[2:-2:2])) + f1(b))
    midpoint = h * np.sum(f1(midpoints))

    print("Q1: Composite Integration Methods")
    print(f"  Trapezoidal Rule: {trapezoidal}")
    print(f"  Simpson's Rule:  {simpson}")
    print(f"  Midpoint Rule:   {midpoint}")

# -------------------------
# Q2. Gaussian Quadrature
# -------------------------

def question2():
    def f2(x):
        return np.log(x) * x

    a = 1
    b = 1.5

    def gauss_quad(n):
        xi, ci = leggauss(n)
        x_mapped = 0.5 * (b - a) * xi + 0.5 * (b + a)
        return 0.5 * (b - a) * np.sum(ci * f2(x_mapped))

    result_n3 = gauss_quad(3)
    result_n4 = gauss_quad(4)
    exact_value, _ = quad(f2, a, b)

    print("\nQ2: Gaussian Quadrature")
    print(f"  n = 3: {result_n3}")
    print(f"  n = 4: {result_n4}")
    print(f"  Exact: {exact_value}")

# -------------------------
# Q3. Double Integration
# -------------------------

def question3():
    def f(x, y):
        return y

    # Simpson's rule
    n = 4
    m = 4
    x = np.linspace(0, np.pi/4, n+1)
    hx = (np.pi/4) / n
    simpson_sum = 0

    for i in range(n + 1):
        xi = x[i]
        y_max = 2 * np.sin(xi) * np.cos(xi)
        hy = y_max / m
        y = np.linspace(0, y_max, m + 1)
        for j in range(m + 1):
            coeff = 1
            coeff *= 1 if (i == 0 or i == n) else (2 if i % 2 == 0 else 4)
            coeff *= 1 if (j == 0 or j == m) else (2 if j % 2 == 0 else 4)
            simpson_sum += coeff * f(xi, y[j])
    simpson_result = (hx / 3) * (hy / 3) * simpson_sum

    # Gaussian quadrature
    xi, ci = leggauss(3)
    eta, cj = leggauss(3)
    gauss_sum = 0
    for i in range(3):
        for j in range(3):
            x_mapped = 0.5 * (np.pi/4) * (xi[i] + 1)
            y_upper = 2 * np.sin(x_mapped) * np.cos(x_mapped)
            y_mapped = 0.5 * y_upper * (eta[j] + 1)
            gauss_sum += ci[i] * cj[j] * f(x_mapped, y_mapped)
    gauss_result = 0.25 * (np.pi/4) * gauss_sum

    # Exact value
    exact, _ = dblquad(f, 0, np.pi/4, lambda x: 0, lambda x: 2*np.sin(x)*np.cos(x))

    print("\nQ3: Double Integration")
    print(f"  Simpson’s Rule (n=4, m=4): {simpson_result}")
    print(f"  Gaussian Quadrature (n=3, m=3): {gauss_result}")
    print(f"  Exact Value: {exact}")

# -------------------------
# Q4. Improper Integrals
# -------------------------

def question4():
    # a) improper integral via x = t²
    def fa(t):
        x = t ** 2
        return 2 * np.sin(x)

    t_a = np.linspace(0, 1, 5)
    ha = (1 - 0) / 4
    sa = fa(t_a[0]) + fa(t_a[-1]) + 4 * np.sum(fa(t_a[1:-1:2])) + 2 * np.sum(fa(t_a[2:-2:2]))
    result_a = ha / 3 * sa

    # b) improper integral via x = 1/t
    def fb(t):
        x = 1 / t
        dxdt = -1 / t ** 2
        return (np.sin(x) / x ** 4) * dxdt

    t_b = np.linspace(0.01, 1, 5)  # Avoid divide-by-zero
    hb = (1 - 0.01) / 4
    sb = fb(t_b[0]) + fb(t_b[-1]) + 4 * np.sum(fb(t_b[1:-1:2])) + 2 * np.sum(fb(t_b[2:-2:2]))
    result_b = hb / 3 * sb

    print("\nQ4: Improper Integrals")
    print(f"  a) ∫₀¹ sin(x)/√x dx ≈ {result_a}")
    print(f"  b) ∫₁^∞ sin(x)/x⁴ dx ≈ {result_b}")

# -------------------------
# Main function
# -------------------------

if __name__ == "__main__":
    question1()
    question2()
    question3()
    question4()
