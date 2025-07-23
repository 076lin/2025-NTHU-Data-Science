# you must use python 3.10
# For linux, you must use download HomeworkFramework.cpython-310-x86_64-linux-gnu.so
# For Mac, you must use download HomeworkFramework.cpython-310-darwin.so
# If above can not work, you can use colab and download HomeworkFramework.cpython-310-x86_64-linux-gnu.so and don't forget to modify output's name.
import numpy as np
from HomeworkFramework import Function


class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)



    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES):
        n = self.dim
        lower = self.lower
        upper = self.upper
        eval_budget = FES

        # CMA-ES 參數
        mu = 4 + int(3 * np.log(n))          # parent size
        lam = 4 * mu                         # offspring size
        sigma = 0.3 * (upper - lower)        # step size (vectorized)
        sigma_scalar = 0.5                   # global scale

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # 初始化
        mean = np.random.uniform(lower, upper, size=n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.identity(n)
        inv_sqrt_C = np.identity(n)

        evals = 0
        best_value = float("inf")
        best_solution = None

        while evals < eval_budget:
            # 產生 λ 個樣本
            z = np.random.randn(lam, n)
            y = z @ np.linalg.cholesky(C).T
            x = mean + sigma_scalar * y * sigma
            x = np.clip(x, lower, upper)

            # 安全 evaluate，每個樣本單獨確認
            fitness = []
            y_valid = []
            x_valid = []

            for xi, yi in zip(x, y):
                if evals >= eval_budget:
                    break
                val = self.f.evaluate(self.target_func, xi)
                if val == "ReachFunctionLimit":
                    break
                fitness.append(float(val))
                x_valid.append(xi)
                y_valid.append(yi)
                evals += 1

            if len(fitness) == 0:
                break  # FES 耗盡，跳出

            fitness = np.array(fitness)
            y_valid = np.array(y_valid)
            x_valid = np.array(x_valid)

            # 更新最佳解
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_value:
                best_value = fitness[min_idx]
                best_solution = x_valid[min_idx]

            # 選前 μ 個做更新
            idx_sorted = np.argsort(fitness)
            y_top = y_valid[idx_sorted[:mu]]
            weighted_y = np.sum(weights[:, None] * y_top, axis=0)

            # 更新 mean
            mean += sigma_scalar * weighted_y * sigma

            # 更新 paths
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ weighted_y)
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * np.sqrt(n)) else 0.0
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * weighted_y

            # 更新協方差矩陣
            # delta_h = (1 - h_sigma) * cc * (2 - cc)
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(w * np.outer(yi, yi) for w, yi in zip(weights, y_top))
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            #   更新步長
            sigma_scalar *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))

            # 更新逆根號矩陣
            try:
                inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C).T)
            except np.linalg.LinAlgError:
                inv_sqrt_C = np.identity(n)

        # 最後輸出結果
        self.optimal_solution[:] = best_solution
        self.optimal_value = best_value



if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
