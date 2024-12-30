import random
import numpy as np
from typing import Dict, List, Set


class WaveOptimizer:
    def __init__(self, order_dict: Dict[str, List[str]], n_waves: int = 100,
                 population_size: int = 100, n_generations: int = 200,
                 mutation_rate: float = 0.1, elite_size: int = 10):
        """
        初始化波次优化器

        Args:
            order_dict: 订单字典，key为订单ID，value为商品列表
            n_waves: 波次数量
            population_size: 种群大小
            n_generations: 迭代代数
            mutation_rate: 变异率
            elite_size: 精英个体数量
        """
        self.order_dict = order_dict
        self.n_waves = n_waves
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.order_ids = list(order_dict.keys())

    def create_individual(self) -> List[int]:
        """
        创建一个个体（染色体），即订单到波次的分配方案
        """
        return [random.randint(0, self.n_waves - 1) for _ in range(len(self.order_ids))]

    def calculate_fitness(self, individual: List[int]) -> float:
        """
        计算适应度（总分拣数的倒数）
        """
        wave_products: List[Set[str]] = [set() for _ in range(self.n_waves)]

        # 统计每个波次中的商品种类
        for order_idx, wave_idx in enumerate(individual):
            order_id = self.order_ids[order_idx]
            products = self.order_dict[order_id]
            wave_products[wave_idx].update(products)

        # 计算总分拣数（所有波次的商品种类数之和）
        total_picking = sum(len(products) for products in wave_products)

        return 1.0 / (total_picking + 1)  # 添加1避免除零错误

    def tournament_selection(self, population: List[List[int]],
                             fitness_scores: List[float]) -> List[int]:
        """
        锦标赛选择
        """
        tournament_size = 3
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[idx] for idx in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        单点交叉
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, individual: List[int]) -> None:
        """
        变异操作
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.n_waves - 1)

    def optimize(self) -> tuple:
        """
        运行遗传算法优化

        Returns:
            tuple: (最优解, 最优解的总分拣数)
        """
        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]

        best_solution = None
        best_fitness = float('-inf')

        for generation in range(self.n_generations):
            # 计算适应度
            fitness_scores = [self.calculate_fitness(ind) for ind in population]

            # 更新最优解
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # 精英保留
            elite = sorted(zip(fitness_scores, population), reverse=True)[:self.elite_size]
            new_population = [ind for _, ind in elite]

            # 生成新一代
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            population = new_population

            if generation % 10 == 0:
                best_picking = self.get_total_picking(best_solution)
                print(f"Generation {generation}: Best total picking = {best_picking}")

        return best_solution, self.get_total_picking(best_solution)

    def get_total_picking(self, solution: List[int]) -> int:
        """
        获取解决方案的总分拣数
        """
        wave_products = [set() for _ in range(self.n_waves)]
        for order_idx, wave_idx in enumerate(solution):
            order_id = self.order_ids[order_idx]
            products = self.order_dict[order_id]
            wave_products[wave_idx].update(products)

        return sum(len(products) for products in wave_products)

    def get_wave_assignments(self, solution: List[int]) -> Dict[int, List[str]]:
        """
        获取波次分配方案

        Returns:
            Dict[int, List[str]]: 波次到订单的映射
        """
        wave_assignments = {i: [] for i in range(self.n_waves)}
        for order_idx, wave_idx in enumerate(solution):
            order_id = self.order_ids[order_idx]
            wave_assignments[wave_idx].append(order_id)
        return wave_assignments


if __name__ == "__main__":
    # 示例数据
    order_dict = {
        "order1": ["product1", "product2"],
        "order2": ["product2", "product3"],
        "order3": ["product1", "product3", "product4"],
        "order4": ["product2", "product4"],
        "order5": ["product1", "product5"]
    }

    # 创建优化器实例
    optimizer = WaveOptimizer(
        order_dict=order_dict,
        n_waves=2,  # 示例中使用2个波次
        population_size=50,
        n_generations=100,
        mutation_rate=0.1,
        elite_size=5
    )

    # 运行优化
    best_solution, total_picking = optimizer.optimize()

    # 获取波次分配方案
    wave_assignments = optimizer.get_wave_assignments(best_solution)

    # 打印结果
    print("\nOptimization Results:")
    print(f"Total picking count: {total_picking}")
    print("\nWave assignments:")
    for wave_idx, orders in wave_assignments.items():
        print(f"Wave {wave_idx}: {orders}")