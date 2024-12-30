import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import time
from tqdm import tqdm

class TabuSearch:
    def __init__(self, order_dict: Dict[str, List[str]],
                 batch_size: int = 16,
                 num_batches: int = 100,
                 max_iterations: int = 1000,
                 tabu_tenure: int = 20):
        """
        初始化禁忌搜索算法

        Args:
            order_dict: 订单字典，key为订单ID，value为商品列表
            batch_size: 每个波次包含的订单数量
            num_batches: 波次总数
            max_iterations: 最大迭代次数
            tabu_tenure: 禁忌长度
        """
        self.order_dict = order_dict
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
        self.best_solution = None
        self.best_cost = float('inf')

    def generate_initial_solution(self) -> List[List[str]]:
        """生成初始解"""
        # 获取所有订单ID
        order_ids = list(self.order_dict.keys())
        random.shuffle(order_ids)

        # 将订单均匀分配到波次中
        solution = []
        for i in range(0, len(order_ids), self.batch_size):
            batch = order_ids[i:i + self.batch_size]
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
        total_items = 0
        for batch in solution:
            total_items += self.calculate_picking_items(batch)
        return total_items

    def get_neighbors(self, solution: List[List[str]]) -> List[Tuple[List[List[str]], Tuple]]:
        """获取邻域解"""
        neighbors = []
        # 随机选择部分批次进行交换以提高效率
        batch_indices = random.sample(range(len(solution)), min(10, len(solution)))

        for i in batch_indices:
            for j in range(i + 1, len(solution)):
                # 在两个批次之间随机选择订单进行交换
                order_i = random.choice(solution[i])
                order_j = random.choice(solution[j])

                # 生成新解
                new_solution = [batch[:] for batch in solution]
                new_solution[i][new_solution[i].index(order_i)] = order_j
                new_solution[j][new_solution[j].index(order_j)] = order_i

                # 记录移动
                move = (order_i, order_j)
                neighbors.append((new_solution, move))

        return neighbors

    def is_tabu(self, move: Tuple) -> bool:
        """检查移动是否在禁忌列表中"""
        return move in self.tabu_list or (move[1], move[0]) in self.tabu_list

    def update_tabu_list(self, move: Tuple):
        """更新禁忌列表"""
        self.tabu_list.append(move)
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)

    def solve(self) -> Tuple[List[List[str]], int]:
        """运行禁忌搜索算法"""
        start_time = time.time()

        # 生成初始解
        current_solution = self.generate_initial_solution()
        current_cost = self.evaluate_solution(current_solution)

        self.best_solution = current_solution
        self.best_cost = current_cost

        # 记录没有改善的迭代次数
        no_improvement = 0

        for iteration in tqdm(range(self.max_iterations)):
            # 获取邻域解
            neighbors = self.get_neighbors(current_solution)

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

            # 更新禁忌列表
            self.update_tabu_list(best_move)

            # 更新最优解
            if current_cost < self.best_cost:
                self.best_solution = current_solution
                self.best_cost = current_cost
                no_improvement = 0
            else:
                no_improvement += 1

            # 如果连续50次迭代没有改善，提前终止
            if no_improvement >= 50:
                break

        end_time = time.time()
        print(f"运行时间: {end_time - start_time:.2f}秒")
        print(f"迭代次数: {iteration + 1}")

        return self.best_solution, self.best_cost


def save_solution(solution: List[List[str]],
                  cost: int,
                  order_dict: Dict[str, List[str]],
                  filename: str):
    """保存结果到文件"""
    with open(filename, 'w', encoding='gb2312') as f:
        f.write("波次序号,订单编号,分拣项数\n")
        for i, batch in enumerate(solution, 1):
            # 计算该波次的分拣项数
            unique_items = set()
            for order_id in batch:
                unique_items.update(order_dict[order_id])
            picking_items = len(unique_items)

            # 写入数据
            f.write(f"{i},{'|'.join(batch)},{picking_items}\n")

        f.write(f"\n总分拣项数: {cost}")

# 使用示例：
# tabu_search = TabuSearch(order_dict)
# best_solution, best_cost = tabu_search.solve()
# save_solution(best_solution, best_cost, order_dict, 'result.csv')