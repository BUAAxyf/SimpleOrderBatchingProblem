from model.GA_Optimizer import GA_Optimizer
from model.ImprovedTS import ImprovedTabuSearch
from model.TabuSearch import TabuSearch, save_solution
from model.WaveOptimizer import WaveOptimizer
from read.Read import Read
from model.ImprovedWaveOptimizer import ImprovedWaveOptimizer

def DEAP_GA_solve(order_dict):
    """
    DEAP框架遗传算法求解
    :return:
    """
    # 参数设置
    num_wave = 1000  # 波次数量
    population_size = 200  # 种群数量
    generations = 1000  # 迭代次数
    cx_prob = 0.8  # 交叉概率
    mut_prob = 0.5  # 变异概率

    # 初始化优化器
    optimizer = GA_Optimizer(order_dict,
                             num_wave=num_wave,
                             population_size=population_size,
                             generations=generations,
                             cx_prob=cx_prob,
                             mut_prob=mut_prob)

    # 运行优化器
    best_individual = optimizer.run_optimizer()

    # 输出最终最优分配
    print("Final best individual (Wave allocation):")
    print(best_individual)

def Wave_GA_solve(order_dict):
    """
    波次分配遗传算法求解
    :return:
    """
    # 创建优化器实例
    optimizer = WaveOptimizer(
        order_dict=order_dict,
        n_waves=100,
        population_size=1000,
        n_generations=1000,
        mutation_rate=0.8,
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

def Improved_Wave_GA_solve(order_dict):
    """
    改进的波次分配遗传算法求解
    :return:
    """
    # 2. 创建优化器实例
    optimizer = ImprovedWaveOptimizer(
        order_dict=order_dict,  # 订单字典
        n_waves=100,  # 波次数量
        population_size=200,  # 种群大小，建议设置为订单数量的2倍
        n_generations=300,  # 迭代代数
        mutation_rate=0.3,  # 变异率
        elite_size=20,  # 精英个体数量
        crossover_rate=0.8,  # 交叉概率
        tournament_size=5,  # 锦标赛选择的参与者数量
        n_processes=None  # 并行进程数，默认使用所有CPU核心
    )

    # 3. 运行优化
    best_solution, final_picking = optimizer.optimize()

    # 4. 获取优化结果
    # 获取波次分配方案
    wave_assignments = optimizer.get_wave_assignments(best_solution)

    # 获取详细统计信息
    stats = optimizer.get_statistics(best_solution)

    # 5. 打印结果
    print(f"\n最终总分拣数: {final_picking}")

    print("\n波次分配详情:")
    for wave_id, orders in wave_assignments.items():
        if orders:  # 只打印有订单的波次
            print(f"波次 {wave_id}: {len(orders)} 个订单 - {orders}")
        else:
            print(f"波次 {wave_id}: 无订单")

    print("\n优化统计信息:")
    print(f"平均每波次分拣数: {stats['avg_picking_per_wave']:.2f}")
    print(f"最大波次分拣数: {stats['max_picking_per_wave']}")
    print(f"最小波次分拣数: {stats['min_picking_per_wave']}")
    print(f"波次分拣数标准差: {stats['std_picking_per_wave']:.2f}")
    print(f"平均每波次订单数: {stats['avg_orders_per_wave']:.2f}")
    print(f"实际使用波次数: {stats['utilized_waves']}")

def TS_solve(order_dict):
    tabu_search = TabuSearch(order_dict)
    best_solution, best_cost = tabu_search.solve()
    save_solution(best_solution, best_cost, order_dict, 'result/TS_result.csv')

def Improved_TS_solve(order_dict):
    tabu_search = ImprovedTabuSearch(order_dict)
    best_solution, best_cost = tabu_search.solve()
    save_solution(best_solution, best_cost, order_dict, 'result/Improved_TS_result.csv')

if __name__ == '__main__':
    # input
    file_path = "data/Order.csv"
    order_dict, product_list, order_list = Read.order_csv(file_path)
    # print(order_dict)

    # solve
    # DEAP_GA_solve(order_dict)
    # Wave_GA_solve(order_dict)
    # Improved_Wave_GA_solve(order_dict)
    # TS_solve(order_dict)
    Improved_TS_solve(order_dict)