import random
from collections import defaultdict
from deap import base, creator, tools, algorithms
import os


class GA_Optimizer:
    def __init__(self, order_dict,
                 num_wave=100,
                 population_size=200,
                 generations=50,
                 cx_prob=0.7,
                 mut_prob=0.2,
                 indpb=0.2):
        """
        初始化优化器
        :param order_dict: 订单字典
        :param num_wave: 进化的周期数
        :param population_size: 每一代的种群大小，决定了参与进化的个体数目
        :param generations: 迭代代数
        :param cx_prob: 交叉操作的概率
        :param mut_prob: 变异操作的概率
        :param indpb: 变异操作中，每个元素被选中的概率
        """
        # 订单字典
        self.order_dict = order_dict
        # 波次数目
        self.num_wave = num_wave
        # 遗传算法参数
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.indpb = indpb

        # 创建遗传算法的适应度和个体
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化问题
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # 初始化工具箱
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)  # 双点交叉
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.indpb)  # 随机变异（交换商品波次）
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择
        self.toolbox.register("evaluate", self.evaluate)

        # 确保结果目录存在
        if not os.path.exists('result'):
            os.makedirs('result')

        # 结果文件路径
        self.result_file = os.path.join('result', 'result.txt')

    def create_individual(self):
        """
        初始化个体，每个商品属于一个波次
        :return: 随机波次分配
        """
        total_products = sum(len(products) for products in self.order_dict.values())
        return [random.randint(0, self.num_wave - 1) for _ in range(total_products)]

    def evaluate(self, individual):
        """
        计算适应度，波次分配的分拣数
        :param individual: 波次分配
        :return: 总分拣数
        """
        wave_dict = defaultdict(set)  # key: 波次, value: 商品种类集合
        product_idx = 0
        # 按照商品分配到波次
        for order_id, product_list in self.order_dict.items():
            for product in product_list:
                wave = individual[product_idx]  # 商品所属的波次
                wave_dict[wave].add(product)  # 将商品加入该波次
                product_idx += 1

        # 计算总分拣数：每个波次的商品种类数
        total_splitting_count = sum(len(products) for products in wave_dict.values())
        return total_splitting_count,  # 返回一个元组，DEAP要求

    def run_optimizer(self):
        """
        运行遗传算法求解最优波次分配
        :return: 最优波次分配
        """
        # 初始化种群
        population = self.toolbox.population(n=self.population_size)

        # 打开文件以追加写入每代信息
        with open(self.result_file, 'w') as result_file:
            result_file.write("Generation, Best Individual Fitness, Best Individual\n")

            # 遗传算法主循环
            for gen in range(self.generations):
                # 评估种群
                fitnesses = list(map(self.toolbox.evaluate, population))
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit

                # 选择、交叉、变异
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))

                # 交叉操作
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.cx_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # 变异操作
                for mutant in offspring:
                    if random.random() < self.mut_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # 评估新个体
                invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(map(self.toolbox.evaluate, invalid_individuals))
                for ind, fit in zip(invalid_individuals, fitnesses):
                    ind.fitness.values = fit

                # 替换种群
                population[:] = offspring

                # 输出当前最优个体
                top_individual = tools.selBest(population, 1)[0]

                # 记录当前代的最优适应度到文件
                result_file.write(f"{gen}, {top_individual.fitness.values[0]}, {top_individual}\n")

                # 输出当前代信息
                # print(f"Generation {gen}: Best individual {top_individual}, Fitness: {top_individual.fitness.values[0]}")
                print(f"Generation {gen}: Fitness: {top_individual.fitness.values[0]}")

        # 最终的最优解
        best_individual = tools.selBest(population, 1)[0]
        return best_individual
