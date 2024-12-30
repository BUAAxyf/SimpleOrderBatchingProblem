import random
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


class ImprovedWaveOptimizer:
    def __init__(self, order_dict: Dict[str, List[str]], n_waves: int = None,
                 population_size: int = 100, n_generations: int = 200,
                 mutation_rate: float = 0.1, elite_size: int = 10,
                 crossover_rate: float = 0.8, tournament_size: int = 5,
                 n_processes: int = None):
        """
        改进版波次优化器，确保每个波次有16个订单

        Args:
            order_dict: 订单字典
            n_waves: 波次数量（将根据订单总数自动计算）
            population_size: 种群大小
            n_generations: 迭代代数
            mutation_rate: 变异率
            elite_size: 精英个体数量
            crossover_rate: 交叉概率
            tournament_size: 锦标赛选择的参与者数量
            n_processes: 并行进程数，默认为CPU核心数
        """
        self.order_dict = order_dict
        # 计算需要的波次数量，确保每波16个订单
        total_orders = len(order_dict)
        self.n_waves = n_waves or (total_orders + 15) // 16
        if total_orders % 16 != 0:
            raise ValueError(f"订单总数（{total_orders}）必须是16的倍数")

        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.order_ids = list(order_dict.keys())
        self.n_processes = n_processes or multiprocessing.cpu_count()

        # 预处理订单数据
        self.preprocess_orders()

    def create_individual(self) -> List[int]:
        """
        创建一个个体，确保每个波次恰好有16个订单
        """
        if random.random() < 0.5:  # 50%概率使用启发式方法
            return self.create_heuristic_individual()
        return self.create_balanced_random_individual()

    def create_balanced_random_individual(self) -> List[int]:
        """
        创建平衡的随机个体，确保每个波次恰好有16个订单
        """
        assignment = []
        for wave in range(self.n_waves):
            assignment.extend([wave] * 16)
        random.shuffle(assignment)
        return assignment

    def create_heuristic_individual(self) -> List[int]:
        """
        使用启发式方法创建个体，确保每个波次恰好有16个订单
        """
        n_orders = len(self.order_ids)
        assignment = [-1] * n_orders
        available_orders = set(range(n_orders))
        orders_per_wave = {i: [] for i in range(self.n_waves)}

        for wave in range(self.n_waves):
            # 如果还有未分配的订单
            while len(orders_per_wave[wave]) < 16 and available_orders:
                if not orders_per_wave[wave]:  # 波次为空，随机选择种子订单
                    seed = random.choice(list(available_orders))
                    orders_per_wave[wave].append(seed)
                    available_orders.remove(seed)
                    assignment[seed] = wave
                else:
                    # 基于相似度选择下一个订单
                    current_orders = orders_per_wave[wave]
                    best_similarity = -1
                    best_order = None

                    for order in available_orders:
                        avg_similarity = np.mean([self.similarity_matrix[order, curr]
                                                  for curr in current_orders])
                        if avg_similarity > best_similarity:
                            best_similarity = avg_similarity
                            best_order = order

                    if best_order is not None:
                        orders_per_wave[wave].append(best_order)
                        available_orders.remove(best_order)
                        assignment[best_order] = wave

        return assignment

    def adaptive_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        修改后的交叉操作，保持每个波次16个订单的平衡
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent1)

        # 随机选择交叉点，必须是16的倍数
        crossover_point = random.randint(1, self.n_waves - 1) * 16

        # 复制前半部分
        wave_counts1 = defaultdict(int)
        wave_counts2 = defaultdict(int)

        # 复制交叉点之前的部分
        for i in range(crossover_point):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
            wave_counts1[parent1[i]] += 1
            wave_counts2[parent2[i]] += 1

        # 填充剩余部分，确保每个波次都有16个订单
        remaining_pos1 = list(range(crossover_point, len(parent1)))
        remaining_pos2 = list(range(crossover_point, len(parent2)))

        for i in range(self.n_waves):
            remaining1 = 16 - wave_counts1[i]
            remaining2 = 16 - wave_counts2[i]

            # 为child1选择合适的位置
            positions1 = remaining_pos1[:remaining1]
            remaining_pos1 = remaining_pos1[remaining1:]
            for pos in positions1:
                child1[pos] = i

            # 为child2选择合适的位置
            positions2 = remaining_pos2[:remaining2]
            remaining_pos2 = remaining_pos2[remaining2:]
            for pos in positions2:
                child2[pos] = i

        return child1, child2

    def adaptive_mutation(self, individual: List[int], generation: int) -> None:
        """
        修改后的变异操作，保持每个波次16个订单的平衡
        """
        current_mutation_rate = self.mutation_rate * (1 - generation / self.n_generations)

        for i in range(len(individual)):
            if random.random() < current_mutation_rate:
                # 找到另一个波次中的一个位置进行交换
                current_wave = individual[i]
                other_wave = random.randint(0, self.n_waves - 1)
                if other_wave == current_wave:
                    continue

                # 找到属于other_wave的一个随机位置
                swap_candidates = [j for j, wave in enumerate(individual) if wave == other_wave]
                if swap_candidates:
                    j = random.choice(swap_candidates)
                    individual[i], individual[j] = individual[j], individual[i]

    def calculate_fitness(self, individual: List[int]) -> float:
        """
        计算适应度，增加对波次订单数量平衡的惩罚项
        """
        wave_products = [set() for _ in range(self.n_waves)]
        wave_orders = [[] for _ in range(self.n_waves)]

        # 统计每个波次的商品和订单
        for order_idx, wave_idx in enumerate(individual):
            order_id = self.order_ids[order_idx]
            products = self.order_dict[order_id]
            wave_products[wave_idx].update(products)
            wave_orders[wave_idx].append(order_id)

        # 检查每个波次是否恰好有16个订单
        for orders in wave_orders:
            if len(orders) != 16:
                return 0.0  # 严重惩罚不满足约束的解

        # 计算总分拣数
        total_picking = sum(len(products) for products in wave_products)

        # 计算波次间商品数量的平衡性
        products_per_wave = [len(products) for products in wave_products]
        balance_penalty = np.std(products_per_wave) if products_per_wave else 0

        # 综合评分（越小越好）
        score = total_picking + balance_penalty * 10

        return 1.0 / (score + 1)

# import random
# import numpy as np
# from typing import Dict, List, Set, Tuple
# from collections import defaultdict
# import time
# from concurrent.futures import ProcessPoolExecutor
# import multiprocessing
#
#
# class ImprovedWaveOptimizer:
#     def __init__(self, order_dict: Dict[str, List[str]], n_waves: int = 100,
#                  population_size: int = 100, n_generations: int = 200,
#                  mutation_rate: float = 0.1, elite_size: int = 10,
#                  crossover_rate: float = 0.8, tournament_size: int = 5,
#                  n_processes: int = None):
#         """
#         改进版波次优化器
#
#         Args:
#             order_dict: 订单字典
#             n_waves: 波次数量
#             population_size: 种群大小
#             n_generations: 迭代代数
#             mutation_rate: 变异率
#             elite_size: 精英个体数量
#             crossover_rate: 交叉概率
#             tournament_size: 锦标赛选择的参与者数量
#             n_processes: 并行进程数，默认为CPU核心数
#         """
#         self.order_dict = order_dict
#         self.n_waves = n_waves
#         self.population_size = population_size
#         self.n_generations = n_generations
#         self.mutation_rate = mutation_rate
#         self.elite_size = elite_size
#         self.crossover_rate = crossover_rate
#         self.tournament_size = tournament_size
#         self.order_ids = list(order_dict.keys())
#         self.n_processes = n_processes or multiprocessing.cpu_count()
#
#         # 预处理订单数据
#         self.preprocess_orders()
#
    def preprocess_orders(self):
        """
        预处理订单数据，建立商品和订单的关联索引
        """
        self.product_to_orders = defaultdict(set)
        self.order_to_products = defaultdict(set)

        for order_id, products in self.order_dict.items():
            self.order_to_products[order_id] = set(products)
            for product in products:
                self.product_to_orders[product].add(order_id)

        # 计算订单相似度矩阵
        self.calculate_order_similarity()

    def calculate_order_similarity(self):
        """
        计算订单间的相似度矩阵，用于启发式分配
        """
        n_orders = len(self.order_ids)
        self.similarity_matrix = np.zeros((n_orders, n_orders))

        for i in range(n_orders):
            for j in range(i + 1, n_orders):
                order1 = self.order_to_products[self.order_ids[i]]
                order2 = self.order_to_products[self.order_ids[j]]

                # 使用Jaccard相似度
                similarity = len(order1 & order2) / len(order1 | order2)
                self.similarity_matrix[i, j] = similarity
                self.similarity_matrix[j, i] = similarity

#     def create_individual(self) -> List[int]:
#         """
#         创建一个个体，使用启发式方法
#         """
#         if random.random() < 0.5:  # 50%概率使用启发式方法
#             return self.create_heuristic_individual()
#         return self.create_random_individual()
#
#     def create_random_individual(self) -> List[int]:
#         """
#         随机创建个体
#         """
#         return [random.randint(0, self.n_waves - 1) for _ in range(len(self.order_ids))]
#
#     def create_heuristic_individual(self) -> List[int]:
#         """
#         使用启发式方法创建个体，将相似订单分配到同一波次
#         """
#         n_orders = len(self.order_ids)
#         assignment = [-1] * n_orders
#         available_orders = set(range(n_orders))
#         current_wave = 0
#
#         while available_orders and current_wave < self.n_waves:
#             # 随机选择一个未分配的订单作为种子
#             seed = random.choice(list(available_orders))
#             available_orders.remove(seed)
#             assignment[seed] = current_wave
#
#             # 根据相似度选择相近的订单
#             similarities = [(i, self.similarity_matrix[seed, i])
#                             for i in available_orders]
#             similarities.sort(key=lambda x: x[1], reverse=True)
#
#             # 将相似订单分配到当前波次
#             for order_idx, sim in similarities:
#                 if sim > 0.3:  # 相似度阈值
#                     assignment[order_idx] = current_wave
#                     available_orders.remove(order_idx)
#
#             current_wave += 1
#
#         # 将剩余订单随机分配
#         for order_idx in available_orders:
#             assignment[order_idx] = random.randint(0, self.n_waves - 1)
#
#         return assignment
#
#     def calculate_fitness(self, individual: List[int]) -> float:
#         """
#         计算适应度，考虑多个目标
#         """
#         wave_products = [set() for _ in range(self.n_waves)]
#         wave_orders = [[] for _ in range(self.n_waves)]
#
#         # 统计每个波次的商品和订单
#         for order_idx, wave_idx in enumerate(individual):
#             order_id = self.order_ids[order_idx]
#             products = self.order_dict[order_id]
#             wave_products[wave_idx].update(products)
#             wave_orders[wave_idx].append(order_id)
#
#         # 计算总分拣数
#         total_picking = sum(len(products) for products in wave_products)
#
#         # 计算波次间的平衡性（订单数量的标准差）
#         orders_per_wave = [len(orders) for orders in wave_orders]
#         balance_penalty = np.std(orders_per_wave) if orders_per_wave else 0
#
#         # 综合评分（越小越好）
#         score = total_picking + balance_penalty * 10
#
#         return 1.0 / (score + 1)
#
#     def parallel_fitness_calculation(self, population: List[List[int]]) -> List[float]:
#         """
#         并行计算种群适应度
#         """
#         with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
#             fitness_scores = list(executor.map(self.calculate_fitness, population))
#         return fitness_scores
#
    def tournament_selection(self, population: List[List[int]],
                             fitness_scores: List[float]) -> List[int]:
        """
        改进的锦标赛选择
        """
        tournament_idx = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[idx] for idx in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
#
#     def adaptive_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
#         """
#         自适应多点交叉
#         """
#         if random.random() > self.crossover_rate:
#             return parent1, parent2
#
#         n_points = random.randint(1, 3)  # 1-3个交叉点
#         points = sorted(random.sample(range(1, len(parent1)), n_points))
#
#         child1, child2 = [], []
#         start = 0
#         swap = True
#
#         for point in points + [len(parent1)]:
#             if swap:
#                 child1.extend(parent1[start:point])
#                 child2.extend(parent2[start:point])
#             else:
#                 child1.extend(parent2[start:point])
#                 child2.extend(parent1[start:point])
#             swap = not swap
#             start = point
#
#         return child1, child2
#
#     def adaptive_mutation(self, individual: List[int], generation: int) -> None:
#         """
#         自适应变异
#         """
#         # 根据进化代数调整变异率
#         current_mutation_rate = self.mutation_rate * (1 - generation / self.n_generations)
#
#         for i in range(len(individual)):
#             if random.random() < current_mutation_rate:
#                 # 局部搜索：倾向于分配到相邻波次
#                 current_wave = individual[i]
#                 new_wave = current_wave + random.choice([-1, 1])
#                 new_wave = max(0, min(new_wave, self.n_waves - 1))
#                 individual[i] = new_wave
#
    def local_search(self, solution: List[int]) -> List[int]:
        """
        局部搜索优化
        """
        improved = True
        best_picking = self.get_total_picking(solution)

        while improved:
            improved = False
            for i in range(len(solution)):
                original_wave = solution[i]
                best_wave = original_wave

                # 尝试将订单移动到其他波次
                for new_wave in range(self.n_waves):
                    if new_wave == original_wave:
                        continue

                    solution[i] = new_wave
                    current_picking = self.get_total_picking(solution)

                    if current_picking < best_picking:
                        best_picking = current_picking
                        best_wave = new_wave
                        improved = True

                solution[i] = best_wave

        return solution

    def optimize(self) -> tuple:
        """
        运行优化算法
        """
        start_time = time.time()

        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]

        best_solution = None
        best_fitness = float('-inf')
        generations_without_improvement = 0

        for generation in range(self.n_generations):
            # 并行计算适应度
            fitness_scores = self.parallel_fitness_calculation(population)

            # 更新最优解
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # 早停机制
            if generations_without_improvement >= 20:
                print(f"Early stopping at generation {generation}")
                break

            # 精英保留
            elite = sorted(zip(fitness_scores, population), reverse=True)[:self.elite_size]
            new_population = [ind for _, ind in elite]

            # 生成新一代
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.adaptive_crossover(parent1, parent2)

                self.adaptive_mutation(child1, generation)
                self.adaptive_mutation(child2, generation)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            if generation % 10 == 0:
                best_picking = self.get_total_picking(best_solution)
                elapsed_time = time.time() - start_time
                print(f"Generation {generation}: Best picking = {best_picking}, "
                      f"Time elapsed: {elapsed_time:.2f}s")

        # 对最优解进行局部搜索优化
        best_solution = self.local_search(best_solution)
        final_picking = self.get_total_picking(best_solution)

        return best_solution, final_picking

    def get_total_picking(self, solution: List[int]) -> int:
        """
        计算总分拣数
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
        """
        wave_assignments = {i: [] for i in range(self.n_waves)}
        for order_idx, wave_idx in enumerate(solution):
            order_id = self.order_ids[order_idx]
            wave_assignments[wave_idx].append(order_id)
        return wave_assignments

    def get_statistics(self, solution: List[int]) -> Dict:
        """
        获取解决方案的详细统计信息
        """
        wave_assignments = self.get_wave_assignments(solution)
        wave_products = [set() for _ in range(self.n_waves)]
        wave_order_counts = []

        for wave_idx, orders in wave_assignments.items():
            wave_order_counts.append(len(orders))
            for order_id in orders:
                products = self.order_dict[order_id]
                wave_products[wave_idx].update(products)

        stats = {
            'total_picking': sum(len(products) for products in wave_products),
            'avg_picking_per_wave': np.mean([len(products) for products in wave_products]),
            'max_picking_per_wave': max(len(products) for products in wave_products),
            'min_picking_per_wave': min(len(products) for products in wave_products),
            'std_picking_per_wave': np.std([len(products) for products in wave_products]),
            'avg_orders_per_wave': np.mean(wave_order_counts),
            'std_orders_per_wave': np.std(wave_order_counts),
            'empty_waves': sum(1 for products in wave_products if not products),
            'utilized_waves': sum(1 for products in wave_products if products)
        }

        return stats
