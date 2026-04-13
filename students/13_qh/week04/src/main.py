import numpy as np
import time
from solvers import AnalyticalSolver, GradientDescentSolver


# =========================
# 数据生成函数
# =========================
def generate_data(N, P, noise_std=0.1):
    np.random.seed(0)

    X = np.random.randn(N, P)

    # 真参数（含截距）
    beta_true = np.random.randn(P + 1)

    y = beta_true[0] + X @ beta_true[1:] + np.random.randn(N) * noise_std

    return X, y, beta_true


# =========================
# MSE
# =========================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# =========================
# 单次实验
# =========================
def run_experiment(N, P):
    print(f"\n===== 实验: N={N}, P={P} =====")

    X, y, beta_true = generate_data(N, P)

    # -------------------------
    # Analytical Solver
    # -------------------------
    start = time.time()
    model1 = AnalyticalSolver().fit(X, y)
    end = time.time()

    y_pred1 = model1.predict(X)

    time1 = end - start
    mse1 = mse(y, y_pred1)

    print("Analytical Solver:")
    print(f"Time: {time1:.4f}s")
    print(f"MSE: {mse1:.6f}")

    # -------------------------
    # Gradient Descent
    # -------------------------
    start = time.time()
    model2 = GradientDescentSolver(lr=0.01, epochs=500)
    model2.fit(X, y)
    end = time.time()

    y_pred2 = model2.predict(X)

    time2 = end - start
    mse2 = mse(y, y_pred2)

    print("\nGradient Descent:")
    print(f"Time: {time2:.4f}s")
    print(f"MSE: {mse2:.6f}")


# =========================
# 主函数
# =========================
if __name__ == "__main__":

    # 实验 A：低维
    run_experiment(N=10000, P=10)

    # 实验 B：高维
    run_experiment(N=10000, P=2000)