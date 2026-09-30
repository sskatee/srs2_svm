import math
from scipy.optimize import bisect as sp_bisect
from scipy.optimize import newton as sp_newton
from scipy.optimize import root_scalar

f   = lambda x: math.exp(x) - 3 * x
df  = lambda x: math.exp(x) - 3
phi = lambda x: math.exp(x) / 3
dphi = lambda x: math.exp(x) / 3

eps1 = 1e-3
eps2 = 1e-5
a, b = 0.0, 1.0

def tabulate(f, a, b, step=0.05):
    print("1. Табулирование f(x) = e^x - 3x на [0; 1]")
    print(f"{'x':>8} {'f(x)':>14}")
    x = a
    while x <= b + 1e-12:
        print(f"{x:>8.2f} {f(x):>14.6f}")
        x += step
    print()

def find_root_interval(f, a, b, step=0.05):
    print("2. Поиск интервала локализации корня")
    x = a
    while x < b:
        if f(x) * f(x + step) < 0:
            print(f"Корень локализован на [{x:.2f}; {x + step:.2f}], "
                  f"f(a)*f(b) = {f(x) * f(x + step):+.3e} < 0")
            print()
            return x, x + step
        x += step
    print("Интервал с корнем не найден")
    return None

def bisection(f, a, b, eps, verbose=True):
    if verbose:
        print(f"Метод бисекции,  eps = {eps:g}")
        print(f"{'k':>3} {'a':>14} {'b':>14} {'c':>14} "
              f"{'f(c)':>14} {'err':>14}")
    n = 0
    while (b - a) / 2 > eps:
        c = (a + b) / 2
        err = (b - a) / 2
        if verbose:
            print(f"{n:>3} {a:>14.9f} {b:>14.9f} {c:>14.9f} "
                  f"{f(c):>14.3e} {err:>14.3e}")
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        n += 1
    x = (a + b) / 2
    if verbose:
        print(f"\n Корень x* = {x:.9f},  итераций = {n},  "
              f"невязка = {abs(f(x)):.3e}\n")
    return x, n

def chords(f, a, b, eps, verbose=True):
    if verbose:
        print(f"Метод хорд,  eps = {eps:g}")
        print(f"{'k':>3} {'a':>14} {'b':>14} {'x_k':>14} "
              f"{'f(x_k)':>14} {'err':>14}")
    x_prev = a
    n = 0
    x = a
    while True:
        x = a - f(a) * (b - a) / (f(b) - f(a))
        err = abs(x - x_prev)
        if verbose:
            print(f"{n:>3} {a:>14.9f} {b:>14.9f} {x:>14.9f} "
                  f"{f(x):>14.3e} {err:>14.3e}")
        n += 1
        if err < eps:
            break
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x
        x_prev = x
    if verbose:
        print(f"\n  Корень x* = {x:.9f},  итераций = {n},  "
              f"невязка = {abs(f(x)):.3e}\n")
    return x, n


def newton(f, df, x0, eps, verbose=True):
    if verbose:
        print(f"Метод Ньютона,  x0 = {x0},  eps = {eps:g}")
        hdr = "f'(x_k)"
        print(f"{'k':>3} {'x_k':>14} {'f(x_k)':>14} {hdr:>14} "
              f"{'x_(k+1)':>14} {'err':>14}")
    x = x0
    n = 0
    while True:
        x_new = x - f(x) / df(x)
        err = abs(x_new - x)
        if verbose:
            print(f"{n:>3} {x:>14.9f} {f(x):>14.3e} {df(x):>14.6f} "
                  f"{x_new:>14.9f} {err:>14.3e}")
        n += 1
        if err < eps:
            x = x_new
            break
        x = x_new
    if verbose:
        print(f"\n  Корень x* = {x:.9f},  итераций = {n},  "
              f"невязка = {abs(f(x)):.3e}\n")
    return x, n

def simple_iteration(phi, x0, eps, verbose=True):
    if verbose:
        print(f"Метод простой итерации,  x0 = {x0},  eps = {eps:g}")
        print(f"{'k':>3} {'x_k':>14} {'phi(x_k)':>14} {'x_(k+1)':>14} "
              f"{'err':>14} {'|f(x_(k+1))|':>14}")
    x = x0
    n = 0
    while True:
        x_new = phi(x)
        err = abs(x_new - x)
        if verbose:
            print(f"{n:>3} {x:>14.9f} {phi(x):>14.9f} {x_new:>14.9f} "
                  f"{err:>14.3e} {abs(f(x_new)):>14.3e}")
        n += 1
        if err < eps:
            x = x_new
            break
        x = x_new
    if verbose:
        print(f"\n  Корень x* = {x:.9f},  итераций = {n},  "
              f"невязка = {abs(f(x)):.3e}\n")
    return x, n

def check_convergence(dphi, a, b, step=0.05):
    print("Проверка условия сходимости |phi'(x)| < 1")
    mx = 0
    x = a
    while x <= b + 1e-12:
        v = abs(dphi(x))
        if v > mx:
            mx = v
        x += step
    print(f"  max |phi'(x)| на [{a}; {b}] = {mx:.6f}")
    print(f"  Условие |phi'(x)| < 1: "
          f"{'Выполнено' if mx < 1 else 'Не выполнено'}\n")

def scipy_solutions(f, df, a, b, eps):
    print("Библиотечные решения SciPy")
    x_bis = sp_bisect(f, a, b, xtol=eps)
    print(f"scipy.optimize.bisect: x* = {x_bis:.9f}, "
          f"невязка = {abs(f(x_bis)):.3e}")
    x_new = sp_newton(f, 0.5, fprime=df, tol=eps)
    print(f"scipy.optimize.newton: x* = {x_new:.9f}, "
          f"невязка = {abs(f(x_new)):.3e}")
    res = root_scalar(f, bracket=[a, b], method='brentq', xtol=eps)
    print(f"scipy.optimize.root_scalar: x* = {res.root:.9f}, "
          f"невязка = {abs(f(res.root)):.3e}, "
          f"итераций = {res.iterations}")
    print()

def summary_table():
    print("Итоговая Таблица Сравнения")
    print(f"{'Метод':<22} {'eps':>8} {'x0':>6} {'x*':>14} "
          f"{'N':>4} {'|f(x*)|':>12}")
    rows = []

    for eps in (eps1, eps2):
        x, n = bisection(f,0.0, 1.0, eps, verbose=False)
        rows.append(("Бисекция", eps, "—", x, n, abs(f(x))))

    for eps in (eps1, eps2):
        x, n = chords(f,0.0, 1.0, eps, verbose=False)
        rows.append(("Хорды", eps, "—", x, n, abs(f(x))))

    for eps in (eps1, eps2):
        x, n = newton(f, df, 0.5, eps, verbose=False)
        rows.append(("Ньютон", eps, "0.5", x, n, abs(f(x))))
    x, n = newton(f, df, 1.0, eps1, verbose=False)
    rows.append(("Ньютон", eps1, "1.0", x, n, abs(f(x))))

    x, n = simple_iteration(phi, 0.5, eps1, verbose=False)
    rows.append(("Простая итерация", eps1, "0.5", x, n, abs(f(x))))
    x, n = simple_iteration(phi, 1.0, eps1, verbose=False)
    rows.append(("Простая итерация", eps1, "1.0", x, n, abs(f(x))))
    x, n = simple_iteration(phi, 0.5, eps2, verbose=False)
    rows.append(("Простая итерация", eps2, "0.5", x, n, abs(f(x))))

    for method, eps, x0, x, n, res in rows:
        print(f"{method:<22} {eps:>8g} {x0:>6} {x:>14.9f} "
              f"{n:>4} {res:>12.3e}")



if __name__ == "__main__":
    tabulate(f, a, b)
    interval = find_root_interval(f, a, b)
    a, b = interval if interval else (a, b)
    check_convergence(dphi, a, b)

    bisection(f, 0.0, 1.0, eps1)
    bisection(f, 0.0, 1.0, eps2)

    chords(f,0.0, 1.0, eps1)
    chords(f, 0.0, 1.0, eps2)

    newton(f, df, 0.5, eps1)
    newton(f, df, 0.5, eps2)
    newton(f, df, 1.0, eps1)

    simple_iteration(phi, 0.5, eps1)
    simple_iteration(phi, 0.5, eps2)
    simple_iteration(phi, 1.0, eps1)

    scipy_solutions(f, df, a, b, eps2)
    summary_table()