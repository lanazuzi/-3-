import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import matplotlib.path as path

n = 2
k = 0

aprox = 15
theta = np.linspace(0, 2 * np.pi, aprox)

accuracy = 500

N_values = [3, 5, 7]
colors_N = ['blue', 'red', 'green']
color_X = 'darkblue'
color_S = 'lightblue'
alpha_values = [0.3, 0.5, 0.7, 0.9]
alpha_colors = ['red', 'orange', 'purple', 'brown']


def MinkowskiSum(current: np.ndarray, other: np.ndarray) -> np.ndarray:
    """ Сумма Минковского """
    sum_vertices = []
    for v1 in current:
        for v2 in other:
            sum_vertices.append(v1 + v2)

    sum_vertices = np.unique(sum_vertices, axis=0)

    return sum_vertices

def plotC(vertices, ax: Axes, color='blue', alpha=0.5, label=None, linestyle='-'):
    """
    Отрисовка выпуклой оболочки множества.
    """
    hull = ConvexHull(vertices)

    closed_vertices = np.vstack([vertices[hull.vertices], vertices[hull.vertices[0]]])

    polygon = Polygon(vertices[hull.vertices], color=color, alpha=0.2, label=label)
    ax.add_patch(polygon)

    ax.plot(closed_vertices[:, 0], closed_vertices[:, 1],
            color=color, alpha=alpha, linestyle=linestyle, linewidth=1.5)

def make_normal_sample(m: np.ndarray, Sigma: np.ndarray, accuracy: int):
    """
    Генерирует выборку из многомерного нормального распределения.

    Параметры:
        m (array_like): вектор средних длины d.
        Sigma (array_like): ковариационная матрица размера d x d.
        n (int): количество точек для генерации.

    Возвращает:
        numpy.ndarray: матрица размера n x d, где каждая строка - точка из N(m, Sigma).
    """
    m = np.asarray(m)
    Sigma = np.asarray(Sigma)
    d = len(m)

    L = np.linalg.cholesky(Sigma)

    Z = np.random.normal(size=(accuracy, d))

    X = Z @ L.T + m

    return X

def MonteKarlo(field: np.ndarray, ax: Axes, points: np.ndarray) -> np.ndarray:
    """
    Генерация выборки из многомерного нормального распределения.

    Parameters:
    mean: вектор средних
    cov: ковариационная матрица
    n: размер выборки

    Returns:
    матрица (dim x n) с точками выборки
    """

    hull = ConvexHull(field)
    p = path.Path(field[hull.vertices])
    inside = p.contains_points(points)

    # ax.scatter(points[inside, 0], points[inside, 1],
    #           color='green', s=10)
    # ax.scatter(points[~inside, 0], points[~inside, 1],
    #           color='red', s=10)

    return sum(inside) / len(points)

def generate_system_matrices():
    S = np.array([[1, 0], [1, 1]])
    S_inv = np.linalg.inv(S)

    r_A, phi_A = 1.5, np.pi/4
    A_real = r_A * np.cos(phi_A)
    A_imag = r_A * np.sin(phi_A)
    lambda_A = np.array([[A_real, A_imag], [-A_imag, A_real]])

    r_B, phi_B = 0.8, np.pi/3
    B_real = r_B * np.cos(phi_B)
    B_imag = r_B * np.sin(phi_B)
    lambda_B = np.array([[B_real, -B_imag], [B_imag, B_real]])

    lambda_C = np.array([[1.2, 0], [0, 0.7]])

    lambda_D = np.array([[0.6, 0], [0, 0.8]])

    lambda_E = np.array([[1.3, 0], [0, 1.4]])

    lambda_F1 = np.array([[1, 1], [0, 1]])
    lambda_F2 = np.array([[-1, 1], [0, -1]])

    lambda_G = np.array([[0.5, 1], [0, 0.6]])

    lambda_H = np.array([[1.3, 1], [0, 1.4]])

    return {
        'A': lambda k: S @ lambda_A @ S_inv,
        'B': lambda k: S @ lambda_B @ S_inv,
        'C': lambda k: S @ lambda_C @ S_inv,
        'D': lambda k: S @ lambda_D @ S_inv,
        'E': lambda k: S @ lambda_E @ S_inv,
        'F1': lambda k: S @ lambda_F1 @ S_inv,
        'F2': lambda k: S @ lambda_F2 @ S_inv,
        'G': lambda k: S @ lambda_G @ S_inv,
        'H': lambda k: S @ lambda_H @ S_inv
    }

U = lambda k: np.array([
    np.array([1, 1]),
    np.array([-1, 1]),
    np.array([-1, -1]),
    np.array([1, -1])]).T

Sigma = lambda k: np.array(
        [[0.025, 0.00001],
        [0.00001, 0.02]])

E = np.array([np.cos(theta), np.sin(theta)])

all_system_matrices = generate_system_matrices()
for mat_name, A_matrices in all_system_matrices.items():
    print(f"\n{'='*50}")
    print(f"Построение графика для матрицы {mat_name}")
    print(f"{'='*50}")

    fig, ax_n = plt.subplots(1, len(N_values), figsize=(18, 6))

    eigenvals = np.linalg.eigvals(A_matrices(0))

    eig_str = []
    for val in eigenvals:
        if np.iscomplex(val):
            eig_str.append(f"{val.real:.2f}{val.imag:+.2f}i")
        else:
            eig_str.append(f"{val:.2f}")

    fig.suptitle(f'Матрица {mat_name} : Собственные значения = {", ".join(eig_str)}',
                 fontsize=14, y=0.98)

    ANk = lambda N, k: np.eye(n) if N <= 0 else A_matrices(N + k - 1) @ ANk(N - 1, k)

    S = np.zeros((1, n))
    SigmaNk = np.zeros((n, n))


    N_prev = 0

    for n_idx, N in enumerate(N_values):

        for j in range(N_prev + 1, N + 1):
            Ajk_U = np.linalg.inv(ANk(j, k)) @ U(k + j - 1)

            S = MinkowskiSum(S, Ajk_U.T)

        ANk_E = np.linalg.inv(ANk(N, k)) @ E

        X = MinkowskiSum(-S, ANk_E.T)
        

        for j in range(N_prev + 1, N + 1):
            SigmaNj = np.linalg.inv(ANk(j, k)) @ Sigma(k + j - 1) @ np.linalg.inv(ANk(j, k)).T
            SigmaNk += SigmaNj

        N_prev = N

        R_max = np.max([np.sqrt(z.T @ np.linalg.inv(SigmaNk) @ z) for z in ANk_E.T])

        ax: Axes = ax_n[n_idx]

        points = make_normal_sample(np.array([0, 0]), SigmaNk, accuracy)

        alpha_max = MonteKarlo(X, ax, points @ ANk(N, k).T)

        print(f"  N={N}: alpha_max = {alpha_max:.4f}, R_max = {R_max:.4f}")

        plotC(X, ax, color=color_X, alpha=0.5,
               label=f'X (сумма управлений с шумами, N={N})', linestyle='-')
        plotC(S, ax, color=color_S, alpha=0.4,
               label=f'S (множество 0-управляемости, N={N})', linestyle='-')

        for a_idx, alpha in enumerate(alpha_values):
            if alpha < alpha_max:
                R = np.sqrt(R_max ** 2 + 2 * np.log(alpha_max / alpha)) - R_max

                ellipsoid = R * np.array([SigmaNk @ p / (np.sqrt(p.T @ SigmaNk @ p)) for p in E.T])

                minkowski_sum_set = MinkowskiSum(S, ellipsoid)

                plotC(minkowski_sum_set, ax, color=alpha_colors[a_idx % 4], alpha=0.3,
                                        label=f'S+Эллипсоид(R={R:.4f}) (α={alpha})',
                                        linestyle='-')


        ax.set_title(f'N = {N}', fontsize=12)

        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=7)

    plt.tight_layout()
    plt.show()