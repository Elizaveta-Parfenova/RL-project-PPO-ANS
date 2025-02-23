import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ------------------------------
# Определение узла (Node)
# ------------------------------
class Node:
    def __init__(self, point):
        self.point = point      # Точка в виде (x, y)
        self.parent = None      # Родительский узел
        self.cost = 0.0         # Стоимость от старта до данного узла

# ------------------------------
# Класс RRTStar
# ------------------------------
class RRTStar:
    def __init__(self, start, goal, grid_map, max_iter=1000, step_size=5, goal_sample_rate=0.2, search_radius=None):
        """
        :param start: Стартовая точка (x, y)
        :param goal: Целевая точка (x, y)
        :param grid_map: 2D numpy-массив, где 1 – препятствие, 0 – свободно
        :param max_iter: максимальное число итераций
        :param step_size: шаг продвижения дерева
        :param goal_sample_rate: вероятность выбрать целевую точку как случайную
        :param search_radius: радиус поиска для переобучения; если None, берётся как step_size * 5
        """
        self.start = Node(start)
        self.goal = Node(goal)
        self.grid_map = grid_map
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.node_list = [self.start]
        self.search_radius = search_radius if search_radius is not None else step_size * 5

    # ------------------------------
    # Генерация случайной точки
    # ------------------------------
    def get_random_point(self):
        if random.random() < self.goal_sample_rate:
            return self.goal.point
        else:
            h, w = self.grid_map.shape  # h – количество строк (y), w – количество столбцов (x)
            return (random.randint(0, w - 1), random.randint(0, h - 1))

    # ------------------------------
    # Поиск ближайшего узла по евклидовой дистанции
    # ------------------------------
    def nearest(self, random_point):
        return min(self.node_list, key=lambda node: np.linalg.norm(np.array(node.point) - np.array(random_point)))

    # ------------------------------
    # Функция "steer": продвигается от from_point к to_point на расстояние step_size
    # ------------------------------
    def steer(self, from_point, to_point):
        from_arr = np.array(from_point, dtype=float)
        to_arr = np.array(to_point, dtype=float)
        direction = to_arr - from_arr
        length = np.linalg.norm(direction)
        if length == 0:
            return from_point
        direction = direction / length
        new_arr = from_arr + self.step_size * direction
        # Округляем до целых, так как координаты в grid_map дискретные
        new_point = (int(round(new_arr[0])), int(round(new_arr[1])))
        return new_point

    # ------------------------------
    # Проверка столкновения для одной точки
    # ------------------------------
    def is_collision(self, point):
        x, y = int(point[0]), int(point[1])
        h, w = self.grid_map.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return True
        # Первый индекс — строка (y), второй — столбец (x)
        return self.grid_map[y, x] == 1

    # ------------------------------
    # Проверка столкновения вдоль линии между p1 и p2
    # ------------------------------
    def is_line_collision(self, p1, p2):
        """
        Дискретизирует отрезок между p1 и p2 и проверяет каждую точку.
        Можно заменить алгоритмом Брезенхэма для повышения точности.
        """
        x1, y1 = p1
        x2, y2 = p2
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance == 0:
            return False
        steps = int(distance)
        for i in range(steps + 1):
            t = i / float(steps)
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            if self.is_collision((x, y)):
                print(f"Collision detected at ({x}, {y})")  # Отладочный вывод
                return True
        return False

    # ------------------------------
    # Поиск узлов, находящихся в радиусе search_radius от new_node
    # ------------------------------
    def get_near_nodes(self, new_node):
        near_nodes = []
        for node in self.node_list:
            if np.linalg.norm(np.array(node.point) - np.array(new_node.point)) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes

    # ------------------------------
    # Выбор наилучшего родителя для new_node из списка near_nodes
    # ------------------------------
    def choose_parent(self, new_node, near_nodes):
        best_cost = float('inf')
        best_parent = None
        for near_node in near_nodes:
            if not self.is_line_collision(near_node.point, new_node.point):
                cost = near_node.cost + np.linalg.norm(np.array(near_node.point) - np.array(new_node.point))
                if cost < best_cost:
                    best_cost = cost
                    best_parent = near_node
        if best_parent is not None:
            new_node.parent = best_parent
            new_node.cost = best_cost

    # ------------------------------
    # Ревайринг: проверка, можно ли улучшить путь до уже существующих узлов через new_node
    # ------------------------------
    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
            potential_cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
            if potential_cost < near_node.cost:
                if not self.is_line_collision(new_node.point, near_node.point):
                    near_node.parent = new_node
                    near_node.cost = potential_cost

    # ------------------------------
    # Основной метод планирования пути
    # ------------------------------
    def plan(self):
        for i in range(self.max_iter):
            random_point = self.get_random_point()
            nearest_node = self.nearest(random_point)
            new_point = self.steer(nearest_node.point, random_point)
            
            # Если новая точка попадает в препятствие или соединение от nearest_node до new_point имеет коллизию — пропускаем итерацию
            if self.is_collision(new_point) or self.is_line_collision(nearest_node.point, new_point):
                continue
            
            new_node = Node(new_point)
            new_node.cost = nearest_node.cost + np.linalg.norm(np.array(nearest_node.point) - np.array(new_point))
            new_node.parent = nearest_node

            # Найти все узлы в окрестности и выбрать наилучшего родителя
            near_nodes = self.get_near_nodes(new_node)
            self.choose_parent(new_node, near_nodes)
            self.node_list.append(new_node)

            # Ревайринг: попытаться улучшить путь для узлов в окрестности через new_node
            self.rewire(new_node, near_nodes)

            # Если новый узел достаточно близок к цели, проверяем соединение с целью
            if np.linalg.norm(np.array(new_node.point) - np.array(self.goal.point)) < self.step_size:
                if not self.is_line_collision(new_node.point, self.goal.point):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(self.goal.point))
                    self.node_list.append(self.goal)
                    return self.extract_path()
        # Если за заданное число итераций путь не найден, возвращаем None
        return None

    # ------------------------------
    # Извлечение пути от цели до старта
    # ------------------------------
    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node.point)
            node = node.parent
        path.reverse()
        return path

# # ------------------------------
# # Пример использования
# # ------------------------------
# if __name__ == "__main__":
#     # Загрузим карту (например, как grayscale-изображение)
#     slam_map = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
    
#     # Преобразуем SLAM-карту в бинарную grid_map:
#     # Например, если препятствия – белые (значения близкие к 255), а свободное пространство – тёмное:
#     threshold = 120
#     # Здесь 1 – препятствие, 0 – свободно.
#     grid_map = np.where(slam_map < threshold, 0, 1)

#     # Задаём стартовую и целевую точки (в координатах пикселей)
#     start = (10, 10)    # (x, y)
#     goal = (grid_map.shape[1] - 10, grid_map.shape[0] - 10)

#     # Создаем экземпляр алгоритма RRT*
#     rrt_star = RRTStar(start, goal, grid_map, max_iter=2000, step_size=10, goal_sample_rate=0.1)

#     path = rrt_star.plan()

#     if path is None:
#         print("Путь не найден")
#     else:
#         print("Найденный путь:")
#         print(path)
        
#         # Визуализация результатов:
#         plt.figure(figsize=(8, 8))
#         plt.imshow(grid_map, cmap='gray')
        
#         # Отрисовываем все узлы дерева
#         for node in rrt_star.node_list:
#             if node.parent is not None:
#                 p1 = node.point
#                 p2 = node.parent.point
#                 plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "-g")
                
#         # Отрисовываем найденный путь
#         path_x = [p[0] for p in path]
#         path_y = [p[1] for p in path]
#         plt.plot(path_x, path_y, "-r", linewidth=2)
        
#         plt.scatter(start[0], start[1], color="blue", s=100, label="Старт")
#         plt.scatter(goal[0], goal[1], color="magenta", s=100, label="Цель")
#         plt.legend()
#         plt.title("RRT*")
#         plt.show()
