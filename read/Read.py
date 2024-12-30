import os

class Read:
    """
    文件、目录读取工具
    """
    def __init__(self):
        pass

    @classmethod
    def data_file(cls, file_path):
        """
        返回路径下的所有文件的路径的列表
        :param file_path:
        :return:
        """
        file_paths = []

        # 检查文件夹是否存在
        if not os.path.isdir(file_path):
            raise ValueError(f"路径 {file_path} 不是一个有效的目录。")

        # 遍历给定目录下的所有文件
        for root, dirs, files in os.walk(file_path):
            for file in files:
                # 获取文件的完整路径并添加到列表中
                full_path = os.path.join(root, file)
                file_paths.append(full_path)

        return file_paths

    @classmethod
    def data_name(cls, file_path):
        """
        返回路径下的所有文件的名称的列表
        :param file_path:
        :return:
        """
        file_names = []

        # 检查文件夹是否存在
        if not os.path.isdir(file_path):
            raise ValueError(f"路径 {file_path} 不是一个有效的目录。")

        # 遍历给定目录下的所有文件
        for root, dirs, files in os.walk(file_path):
            for file in files:
                # 获取文件的名称并添加到列表中
                file_name = os.path.basename(file)
                file_names.append(file_name)

        return file_names

    @classmethod
    def order_csv(cls, file_path):
        """
        read csv file of order info
        :param file_path:
        :return: a dict of {order: list of products}, a list of product_id and a list of order_id
        """
        order_dict = {}
        order_id_list = []
        product_id_list = []
        with open(file_path, 'r') as lines:
            # skip header
            next(lines)

            for line in lines:
                if not line:
                    continue
                # print(line)

                order_id, products = line.split(',') # split order_id and product_list
                product_list = products.split('|') # split product_list
                # print(order_id, product_list)

                product_list[-1] = product_list[-1][:-1]
                # print(order_id, product_list)

                order_dict[order_id] = product_list
        for order_id, product_list in order_dict.items():
            if order_id not in order_id_list:
                order_id_list.append(order_id)

            for product_id in product_list:
                if product_id not in product_id_list:
                    product_id_list.append(product_id)

        return order_dict, product_id_list, order_id_list

if __name__ == '__main__':
    # 测试
    path = "..\\data"
    file_list = Read.data_file(path)
    # print(Read.data_file(path))

    file = file_list[0]
    orderDict, product_id_list, order_id_list = Read.order_csv(file)
    print("orderDict: ", orderDict)
    print("product_id_list: ", product_id_list)
    print("order_id_list: ", order_id_list)