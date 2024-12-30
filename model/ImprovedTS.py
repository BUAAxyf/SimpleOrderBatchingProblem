import random
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import time
import numpy as np
from tqdm import tqdm

class ImprovedTabuSearch:
    def __init__(self, order_dict: Dict[str, List[str]],
                 batch_size: int = 16,
                 num_batches: int = 100,
                 max_iterations: int = 1500,
                 min_tabu_tenure: int = 10,
                 max_tabu_tenure: int = 30,
                 diversification_threshold: int = 100):
        """
        初始化优化后的禁忌搜索算法

        Args:
            order_dict: 订单字典，key为订单ID，value为商品列表
            batch_size: 每个波次包含的订单数量
            num_batches: 波次总数
            max_iterations: 最大迭代次数
            min_tabu_tenure: 最小禁忌期限
            max_tabu_tenure: 最大禁忌期限
            diversification_threshold: 触发多样化策略的阈值
        """
        self.order_dict = order_dict
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.max_iterations = max_iterations
        self.min_tabu_tenure = min_tabu_tenure
        self.max_tabu_tenure = max_tabu_tenure
        self.diversification_threshold = diversification_threshold

        self.tabu_list = {}  # 改用字典存储禁忌移动及其剩余期限
        self.frequency_matrix = defaultdict(int)  # 记录移动频率
        self.best_solution = None
        self.best_cost = float('inf')

        # 预处理订单相似度
        self.similarity_matrix = self._calculate_order_similarity()

    def _calculate_order_similarity(self) -> Dict[Tuple[str, str], float]:
        """计算订单间的相似度"""
        similarity = {}
        orders = list(self.order_dict.keys())

        for i, order1 in enumerate(orders):
            items1 = set(self.order_dict[order1])
            for order2 in orders[i + 1:]:
                items2 = set(self.order_dict[order2])
                # 使用Jaccard相似度
                similarity[(order1, order2)] = len(items1 & items2) / len(items1 | items2)
                similarity[(order2, order1)] = similarity[(order1, order2)]

        return similarity

    def _get_order_similarity(self, order1: str, order2: str) -> float:
        """获取两个订单的相似度"""
        return self.similarity_matrix.get((order1, order2), 0)

    def generate_initial_solution(self) -> List[List[str]]:
        """生成优化的初始解：基于订单相似度的贪心构造"""
        orders = list(self.order_dict.keys())
        solution = []
        remaining_orders = set(orders)

        while remaining_orders:
            # 开始新的批次
            batch = []
            if not batch:
                # 随机选择第一个订单
                first_order = random.choice(list(remaining_orders))
                batch.append(first_order)
                remaining_orders.remove(first_order)

            # 基于相似度填充批次
            while len(batch) < self.batch_size and remaining_orders:
                best_similarity = -1
                best_order = None

                # 计算当前批次与剩余订单的平均相似度
                for order in remaining_orders:
                    avg_similarity = sum(self._get_order_similarity(order, o) for o in batch) / len(batch)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_order = order

                if best_order:
                    batch.append(best_order)
                    remaining_orders.remove(best_order)
                else:
                    break

            solution.append(batch)

        return solution

    def calculate_picking_items(self, batch: List[str]) -> int:
        """计算单个波次的分拣项数"""
        unique_items = set()
        for order_id in batch:
            unique_items.update(self.order_dict[order_id])
        return len(unique_items)

    def evaluate_solution(self, solution: List[List[str]]) -> int:
        """评估解的总分拣项数"""
        return sum(self.calculate_picking_items(batch) for batch in solution)

    def get_neighbors(self, solution: List[List[str]], iteration: int) -> List[Tuple[List[List[str]], Tuple]]:
        """获取改进的邻域解"""
        neighbors = []

        # 动态调整搜索强度
        search_intensity = max(5, min(20, 20 - iteration // 100))
        batch_indices = random.sample(range(len(solution)), min(search_intensity, len(solution)))

        for i in batch_indices:
            for j in range(len(solution)):
                if i == j:
                    continue

                # 选择相似度最低的订单进行交换
                batch_i = solution[i]
                batch_j = solution[j]

                # 计算批次内订单的平均相似度
                avg_sim_i = self._calculate_batch_similarity(batch_i)
                avg_sim_j = self._calculate_batch_similarity(batch_j)

                # 选择相似度较低的订单进行交换
                order_i = self._select_dissimilar_order(batch_i)
                order_j = self._select_dissimilar_order(batch_j)

                # 生成新解
                new_solution = [batch[:] for batch in solution]
                new_solution[i][new_solution[i].index(order_i)] = order_j
                new_solution[j][new_solution[j].index(order_j)] = order_i

                # 记录移动
                move = (order_i, order_j)
                neighbors.append((new_solution, move))

        return neighbors

    def _calculate_batch_similarity(self, batch: List[str]) -> float:
        """计算批次内订单的平均相似度"""
        if len(batch) < 2:
            return 0

        similarities = []
        for i, order1 in enumerate(batch):
            for order2 in batch[i + 1:]:
                similarities.append(self._get_order_similarity(order1, order2))

        return sum(similarities) / len(similarities) if similarities else 0

    def _select_dissimilar_order(self, batch: List[str]) -> str:
        """选择批次中相似度最低的订单"""
        min_similarity = float('inf')
        selected_order = random.choice(batch)

        for order in batch:
            avg_similarity = sum(self._get_order_similarity(order, o) for o in batch if o != order) / (len(batch) - 1)
            if avg_similarity < min_similarity:
                min_similarity = avg_similarity
                selected_order = order

        return selected_order

    def _adjust_tabu_tenure(self, improvement: bool):
        """动态调整禁忌期限"""
        if improvement:
            self.current_tabu_tenure = max(self.min_tabu_tenure,
                                           self.current_tabu_tenure - 1)
        else:
            self.current_tabu_tenure = min(self.max_tabu_tenure,
                                           self.current_tabu_tenure + 1)

    def is_tabu(self, move: Tuple) -> bool:
        """检查移动是否在禁忌列表中"""
        return move in self.tabu_list or (move[1], move[0]) in self.tabu_list

    def update_tabu_list(self, move: Tuple):
        """更新禁忌列表"""
        self.tabu_list[move] = self.current_tabu_tenure
        self.frequency_matrix[move] += 1

        # 更新禁忌期限
        keys_to_remove = []
        for key in self.tabu_list:
            self.tabu_list[key] -= 1
            if self.tabu_list[key] <= 0:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.tabu_list[key]

    def apply_diversification(self, solution: List[List[str]]) -> List[List[str]]:
        """应用多样化策略"""
        # 找出使用频率最高的移动
        frequent_moves = sorted(self.frequency_matrix.items(),
                                key=lambda x: x[1],
                                reverse=True)[:10]

        # 尝试避免这些频繁移动
        new_solution = [batch[:] for batch in solution]
        for (order1, order2), _ in frequent_moves:
            # 在不同批次中找到这些订单
            batch1 = None
            batch2 = None
            for i, batch in enumerate(new_solution):
                if order1 in batch:
                    batch1 = i
                if order2 in batch:
                    batch2 = i

            if batch1 is not None and batch2 is not None and batch1 != batch2:
                # 随机选择其他订单进行交换
                other_order1 = random.choice([o for o in new_solution[batch1] if o != order1])
                other_order2 = random.choice([o for o in new_solution[batch2] if o != order2])

                # 交换订单
                new_solution[batch1][new_solution[batch1].index(order1)] = other_order2
                new_solution[batch2][new_solution[batch2].index(other_order2)] = order1

        return new_solution

    def local_search(self, solution: List[List[str]]) -> List[List[str]]:
        """应用局部搜索优化"""
        improved = True
        current_solution = [batch[:] for batch in solution]

        while improved:
            improved = False
            current_cost = self.evaluate_solution(current_solution)

            # 尝试在每个批次内部重新排列订单
            for i in range(len(current_solution)):
                batch = current_solution[i]
                best_arrangement = batch
                best_cost = self.calculate_picking_items(batch)

                # 尝试不同的排列
                for _ in range(min(20, len(batch))):
                    new_arrangement = batch[:]
                    random.shuffle(new_arrangement)
                    new_cost = self.calculate_picking_items(new_arrangement)

                    if new_cost < best_cost:
                        best_arrangement = new_arrangement
                        best_cost = new_cost
                        improved = True

                current_solution[i] = best_arrangement

        return current_solution

    def solve(self) -> Tuple[List[List[str]], int]:
        """运行优化后的禁忌搜索算法"""
        start_time = time.time()

        # 生成初始解
        current_solution = self.generate_initial_solution()
        current_cost = self.evaluate_solution(current_solution)

        self.best_solution = current_solution
        self.best_cost = current_cost

        # 初始化当前禁忌期限
        self.current_tabu_tenure = self.min_tabu_tenure

        # 记录没有改善的迭代次数
        no_improvement = 0

        for iteration in tqdm(range(self.max_iterations)):
            # 获取邻域解
            neighbors = self.get_neighbors(current_solution, iteration)

            # 找到最佳非禁忌移动或满足特赦规则的移动
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor, move in neighbors:
                neighbor_cost = self.evaluate_solution(neighbor)

                if (not self.is_tabu(move) and neighbor_cost < best_neighbor_cost) or \
                        (neighbor_cost < self.best_cost):  # 特赦规则
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_move = move

            if best_neighbor is None:
                continue

            # 更新当前解
            current_solution = best_neighbor
            current_cost = best_neighbor_cost

            # 更新禁忌列表并调整禁忌期限
            self.update_tabu_list(best_move)

            # 更新最优解
            if current_cost < self.best_cost:
                self.best_solution = current_solution
                self.best_cost = current_cost
                no_improvement = 0
                self._adjust_tabu_tenure(True)
            else:
                no_improvement += 1
                self._adjust_tabu_tenure(False)

            # 应用多样化策略
            if no_improvement >= self.diversification_threshold:
                current_solution = self.apply_diversification(current_solution)
                current_cost = self.evaluate_solution(current_solution)
                no_improvement = 0

            # 定期应用局部搜索
            if iteration % 50 == 0:
                current_solution = self.local_search(current_solution)
                current_cost = self.evaluate_solution(current_solution)

            # 如果连续200次迭代没有改善，提前终止
            if no_improvement >= 200:
                break

        # 最终优化
        self.best_solution = self.local_search(self.best_solution)
        self.best_cost = self.evaluate_solution(self.best_solution)

        end_time = time.time()
        print(f"运行时间: {end_time - start_time:.2f}秒")
        print(f"迭代次数: {iteration + 1}")
        print(f"最终总分拣项数: {self.best_cost}")

        return self.best_solution, self.best_cost


# def save_solution(solution: List[List[str]],
#                   cost: int,
#                   order_dict: Dict[str, List[str]],
#                   filename: str):
#     """保存结果到文件"""