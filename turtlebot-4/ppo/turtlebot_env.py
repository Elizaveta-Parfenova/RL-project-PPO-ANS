import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
import math
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import cv2
import collections
from my_turtlebot_package.rrt_star import RRTStar
import matplotlib.pyplot as plt


def slam_to_grid_map(slam_map, threshold=128):

    grid_map = np.where(slam_map < threshold, 1, 0)  
    num_obstacles = np.count_nonzero(grid_map == 1)

    # print(num_obstacles)
    
    # Визуализация grid_map
    # plt.figure(figsize=(8, 8))
    # plt.imshow(grid_map, cmap='gray')
    # plt.title(f'Grid Map с порогом {threshold}')
    # plt.axis('off')
    # plt.show()
    
    return grid_map

def world_to_map(world_coords, resolution, origin, map_offset, map_shape):
    """
    Преобразует мировые координаты в пиксельные, учитывая смещение
    и возможный переворот карты SLAM.

    Параметры:
      world_coords: (x_world, y_world) – мировые координаты
      resolution: масштаб (размер одного пикселя в мировых единицах)
      origin: мировые координаты начала карты (нижний левый угол SLAM)
      map_offset: (offset_x, offset_y) – сдвиг, чтобы центрировать карту
      map_shape: (map_height, map_width) – размеры карты в пикселях
    
    Возвращает:
      (x_map, y_map): координаты в пиксельной системе
    """
    x_world, y_world = world_coords

    # Перевод в пиксельные координаты
    x_map = int((x_world - origin[0]) / resolution) + map_offset[0]
    y_map = int((y_world - origin[1]) / resolution) + map_offset[1]

    # Переворачиваем Y, если SLAM-карта инвертирована
    y_map = map_shape[0] - y_map - 1

    # Ограничиваем координаты
    x_map = max(0, min(x_map, map_shape[1] - 1))
    y_map = max(0, min(y_map, map_shape[0] - 1))

    return (x_map, y_map)

def path(state, goal, grid_map, map_resolution = 0.05, map_origin = (-4.86, -7.36)):
    
    state_world = state # Текущая позиция в мировых координатах
    goal_world = goal  # Цель в мировых координатах

    map_offset = (45, 15)  # Смещение координат
    map_shape = grid_map.shape  # (высота, ширина) карты

    state_pixel = world_to_map(state_world, map_resolution, map_origin, map_offset, map_shape)
    goal_pixel = world_to_map(goal_world, map_resolution, map_origin, map_offset, map_shape)
    # print(goal_pixel)

    # print(state_pixel)
    # print(goal_pixel)

    rrt_star = RRTStar(state_pixel, goal_pixel, grid_map)
    optimal_path = rrt_star.plan()
    # print(optimal_path)

    # optimal_path = [map_to_world(p, map_resolution, map_origin) for p in optimal_path]

    if optimal_path is None:
        print("Путь не найден")
    else:
        print("Найденный путь:")
        print(optimal_path)
        
        # Визуализация результатов:
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_map, cmap='gray')
        
        # Отрисовываем все узлы дерева
        for node in rrt_star.node_list:
            if node.parent is not None:
                p1 = node.point
                p2 = node.parent.point
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "-g")
                
        # Отрисовываем найденный путь
        path_x = [p[0] for p in optimal_path]
        path_y = [p[1] for p in optimal_path]
        plt.plot(path_x, path_y, "-r", linewidth=2)
        
        plt.scatter(state_pixel[0], state_pixel[1], color="blue", s=100, label="Старт")
        plt.scatter(goal_pixel[0], goal_pixel[1], color="magenta", s=100, label="Цель")
        plt.legend()
        plt.title("RRT*")
        plt.show()
    return optimal_path


# def generate_potential_field(grid_map, goal, path_points, repulsive_scale=5.0, attractive_scale=0.1, repulsive_threshold=0.5):
#     height, width = grid_map.shape
#     potential_field = np.zeros((height, width))

#     obstacles = []
#     for x in range(height):
#         for y in range(width):
#             if grid_map[x, y] == 1:  # Если клетка не пустая (например, препятствие имеет значение > 0)
#                 obstacles.append((x, y))

#     for x in range(height):
#         for y in range(width):
#             attractive_potential = 0
#             repulsive_potential = 0
            
#             # Attractive potential
#             distance_to_goal = np.linalg.norm(np.array([y, x]) - np.array(goal))
#             attractive_potential = attractive_scale * distance_to_goal
            
#             if path_points:
#                 for point in path_points:
#                     distance_to_point = np.linalg.norm(np.array([y, x]) - np.array(point))
#                     attractive_potential += attractive_scale * distance_to_point

#             # Repulsive potential
#             for obs in obstacles:
#                 distance_to_obstacle = np.linalg.norm(np.array([x, y]) - np.array(obs))
#                 if distance_to_obstacle < repulsive_threshold:
#                     if distance_to_obstacle == 0:
#                         distance_to_obstacle = 0.1  # Avoid division by zero
#                     repulsive_potential += repulsive_scale * (1.0 / distance_to_obstacle - 1.0 / repulsive_threshold) ** 2
            
#             potential_field[x, y] = attractive_potential + repulsive_potential
    
#     return potential_field

def generate_potential_field(grid_map, goal, path_points, k_att=0.5, k_rep=5.0, d0=1.1, scale = 0.05):
    """
    Улучшенная генерация потенциального поля:
    - Притягивающий потенциал (quadratic)
    - Отталкивающий потенциал (logarithmic attenuation)
    - Промежуточные точки маршрута усиливают притягивающий потенциал
    """
    height, width = grid_map.shape
    y_coords, x_coords = np.indices(grid_map.shape)

    # Притягивающее поле (quadratic attraction)
    dx = x_coords - goal[0]
    dy = y_coords - goal[1]
    att_field = -0.5 * k_att * np.exp(-scale*np.sqrt(dx**2 + dy**2))

    # Дополнительное притяжение к промежуточным точкам
    att_points = np.zeros_like(grid_map, dtype=np.float32)
    for pt in path_points:
        dx_pt = x_coords - pt[0]
        dy_pt = y_coords - pt[1]
        att_points += -0.2 * k_att * np.exp(-scale*np.sqrt(dx_pt**2 + dy_pt**2))

    # Отталкивающее поле (logarithmic attenuation)
    rep_field = np.zeros_like(grid_map, dtype=np.float64)
    obstacles = np.argwhere(grid_map == 1)

    for (y, x) in obstacles:
        dist_map = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        mask = (dist_map < d0) & (dist_map > 0)
        rep_field[mask] += k_rep * np.log1p(1.0 / (dist_map[mask] + 1e-6))

    # Итоговое поле
    field = att_field + att_points + rep_field

    return field


class TurtleBotEnv(Node, gym.Env):
    def __init__(self):
        super().__init__('turtlebot_env')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_laser = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.subscription_camera = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        
        self.bridge = CvBridge()
        self.camera_obstacle_detected = False
        self.lidar_obstacle_detected = False
        
        self.target_x = -2.0
        self.target_y = -6.0
        self.goal = [self.target_x, self.target_y]
        
        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.state_pose = [-2.0, -0.5]

        slam_map = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
        self.grid_map = slam_to_grid_map(slam_map)

        self.optimal_path = path(self.state_pose, self.goal, self.grid_map)
        self.potential_field = generate_potential_field(self.grid_map, world_to_map(self.goal, 0.05, (-4.86, -7.36), (45, 15), self.grid_map.shape), self.optimal_path)
        self.show_potential_field() 
        self.prev_potential = 0 
        self.prev_x = None  # Предыдущая координата X
        self.prev_y = None  # Предыдущая координата Y
        

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacles = []
        self.prev_distance = None
        self.past_distance = 0
        self.max_steps = 5000
        self.steps = 0 
        
        self.action_space = spaces.Discrete(3)  
        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0, -np.pi, 0.0]), 
                                            high=np.array([10.0, 10.0, np.pi, 12.0]), 
                                            shape=(4,), dtype=np.float32)
        
        self.timer = self.create_timer(0.1, self._timer_callback)


    def _timer_callback(self):
        pass 

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # def scan_callback(self, msg):
        # self.obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max else msg.range_max for r in msg.ranges]
    
    def scan_callback(self, msg):
        raw_obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max 
                        else msg.range_max for r in msg.ranges]
        
        self.obstacles = raw_obstacles
        min_obstacle_dist = min(raw_obstacles) if raw_obstacles else float('inf')
        
        # Конвертация координат
        current_x, current_y = world_to_map(
            (self.current_x, self.current_y),
            resolution=0.05,
            origin=(-4.86, -7.36),
            map_offset=(45, 15),
            map_shape=self.grid_map.shape
        )

        # Получаем значение поля
        potential_value = self.potential_field[current_y, current_x]
        # print(f"Potential: {potential_value:.2f}, Min dist: {min_obstacle_dist:.2f}")
        
        # Логика определения препятствий
        self.lidar_obstacle_detected = (
            (min_obstacle_dist < 0.2) or    # Лидар обнаружил близкое препятствие
            (potential_value > 0.0)          # Высокий потенциал в опасной зоне
        )
    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if cv_image is not None:
                self.camera_obstacle_detected = self.process_camera_image(cv_image)
            else:
                self.camera_obstacle_detected = False
            # print(self.camera_obstacle_detected)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            self.camera_obstacle_detected = False
    
    def show_potential_field(self):

        goal_pixel = world_to_map(self.goal, resolution=0.05, origin=(-4.86, -7.36), map_offset=(45, 15),map_shape=self.grid_map.shape)
        plt.figure(figsize=(10, 8))
        plt.imshow(self.potential_field, cmap='jet')
        plt.colorbar(label='Potential')
        plt.scatter(goal_pixel[0], goal_pixel[1], c='green', s=200, marker='*', label='Goal')
        plt.title("Potential Field Visualization")
        plt.legend()
        plt.show()

    def process_camera_image(self, cv_image):
   
        # Преобразование изображения в формат для обработки
        pixel_values = cv_image.reshape((-1, 3))  # Преобразование в список пикселей
        pixel_values = np.float32(pixel_values)

        # Применение K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 2  # Количество кластеров (фон и препятствие)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Преобразование обратно в изображение
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(cv_image.shape)

        # Анализ кластеров
        obstacle_detected = np.count_nonzero(labels == 1) > 210000 # Если пикселей кластеров > чего-то, то препятствие обнаружено
        # print(obstacle_detected)
        return obstacle_detected
    
    def get_next_state(self, state, action):
        """
        Предсказывает следующее состояние на основе текущего состояния, действия и переданного угла.

        :param state: текущее состояние [x, y, min_obstacle_dist] (без угла)
        :param action: выбранное действие (0 - поворот вправо, 1 - движение вперёд, 2 - поворот влево)
        :param angle: текущий угол робота (передаётся отдельно)
        :return: следующее состояние [next_x, next_y, next_min_obstacle_dist], next_angle
        """
        current_x, current_y, _, min_obstacle_dist = state.squeeze()  # Распаковываем текущее состояние
        next_x, next_y = current_x, current_y  # По умолчанию остаются неизменными
        next_angle = self.current_yaw # Начинаем с текущего угла

        # print(next_angle)
        # Определяем действие
        if action == 0:  # Поворот вправо
            next_angle -= 0.5  
        elif action == 1:  # Движение вперёд
            next_x = current_x + np.cos(self.current_yaw) * 0.2  # Двигаемся в направлении угла
            next_y = current_y + np.sin(self.current_yaw) * 0.2
        elif action == 2:  
            next_angle += 0.5  # Увеличиваем угол

        # Ограничиваем угол в диапазоне [-π, π]
        next_angle = (next_angle + np.pi) % (2 * np.pi) - np.pi

        # min_obstacle_dist остаётся прежним (или можно пересчитывать)
        return np.array([next_x, next_y, next_angle, min_obstacle_dist], dtype=np.float32)

    def compute_potential_reward(self, state, goal, intermediate_points, obstacle_detected, k_att=0.5, k_rep=5.0, d0=1.2, lam=0.5):
        current_x, current_y, _, min_obstacle_dist = state

        # Преобразуем координаты в пиксельные
        current_x, current_y = world_to_map(
            (current_x, current_y),
            resolution=0.05,
            origin=(-4.86, -7.36),
            map_offset=(45, 15),
            map_shape=self.grid_map.shape
        )

        goal_x, goal_y = world_to_map(goal, 0.05, (-4.86, -7.36), (45, 15), self.grid_map.shape)

        # Потенциал в текущей точке
        potential_value = self.potential_field[current_y, current_x]  # поменял индексы
        delta_potential = self.prev_potential - potential_value  # уменьшение потенциала - хорошо
        R_potential = max(-delta_potential, -1)  # если двигаемся в правильном направлении - награда, иначе штраф

        self.prev_potential = potential_value  # обновляем предыдущее значение

        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = current_x, current_y

        # Промежуточные точки
        if intermediate_points:
            nearest_intermediate = min(intermediate_points, key=lambda p: np.linalg.norm([current_x - p[0], current_y - p[1]]))
            prev_dist = np.linalg.norm([self.prev_x - nearest_intermediate[0], self.prev_y - nearest_intermediate[1]])
            curr_dist = np.linalg.norm([current_x - nearest_intermediate[0], current_y - nearest_intermediate[1]])
            R_intermediate = k_att * (prev_dist - curr_dist)  # если приблизился - награда, если удалился - штраф
        else:
            R_intermediate = 0
        
        self.prev_x, self.prev_y = current_x, current_y

        # Градиент потенциального поля
        grad_y, grad_x = np.gradient(self.potential_field)
        local_grad_x = grad_x[current_y, current_x]
        local_grad_y = grad_y[current_y, current_x]
        grad_norm = np.hypot(local_grad_x, local_grad_y)
        grad_reward = -lam * grad_norm  # Чем выше градиент, тем больше штраф

        # Отталкивающее поле (штраф за близость к препятствию)
        if obstacle_detected and min_obstacle_dist > 0:
            R_repulsive = -k_rep * max(0, (1 / min_obstacle_dist - 1 / d0)) ** 2 if min_obstacle_dist < d0 else 0
        else:
            R_repulsive = 0

        # Итоговая награда
        total_reward = R_potential + R_intermediate + grad_reward + R_repulsive

        # Обновляем предыдущее положение
        self.prev_x, self.prev_y = current_x, current_y

        return total_reward


    def step(self, action):
        cmd_msg = Twist()
        if action == 0:
            cmd_msg.angular.z = 0.5  
        elif action == 1:
            cmd_msg.linear.x = 0.2  
        elif action == 2:
            cmd_msg.angular.z = -0.5  
        
        rclpy.spin_once(self, timeout_sec=0.1) 
        self.publisher_.publish(cmd_msg)
    
        self.steps += 1
        
        # print(self.obstacles)
        distance = math.sqrt((self.target_x - self.current_x) ** 2 + (self.target_y - self.current_y) ** 2)
        angle_to_goal = math.atan2(self.target_y - self.current_y, self.target_x - self.current_x)
        angle_diff = (angle_to_goal - self.current_yaw + np.pi) % (2 * np.pi) - np.pi

        min_obstacle_dist = min(self.obstacles) if self.obstacles else float('inf')

        obstacle_detected = self.lidar_obstacle_detected or self.camera_obstacle_detected
        state = np.array([self.current_x, self.current_y, angle_diff, min_obstacle_dist])

        # distance_rate = (self.past_distance - distance)
        # print(min_obstacle_dist)
        reward = self.compute_potential_reward(state, self.goal, self.optimal_path, obstacle_detected)
        # reward += 50.0 * distance_rate
        # self.past_distance = distance

        done = False
        if obstacle_detected:
            reward -= 100
            done = False
        elif distance < 0.2:
            reward += 120 
            done = True
        elif self.steps >= self.max_steps:
            reward -= 100
            done = True
        else:
            done = False
        
        # reward = reward / 100.0 
        # print(state)

        return state, reward, done, {}

    def reset(self):
    # Остановить движение
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0  
        cmd_msg.angular.z = 0.0
        self.publisher_.publish(cmd_msg)  
        rclpy.spin_once(self, timeout_sec=0.1) 
    
        # Использовать reset_simulation для физического сброса в Gazebo
        client = self.create_client(Empty, '/reset_simulation')
        request = Empty.Request()
        if client.wait_for_service(timeout_sec=1.0):
            client.call_async(request)
        else:
            self.get_logger().warn('Gazebo reset service not available!')

    # Сбросить внутренние переменные
        self.current_x = -2.0
        self.current_y = -0.5
        self.current_yaw = 0.0
        self.steps = 0
        self.prev_distance = None
        self.obstacles = []
        self.camera_obstacle_detected = False
        return np.array([self.current_x, self.current_y, 0.0, 0.0])  
 
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)