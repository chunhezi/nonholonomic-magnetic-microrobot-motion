import numpy as np
import pickle
import model
import scipy.sparse as sparse
# import osqp
import scipy.linalg as sl
import math
import cvxpy as cp
from simulation import theta
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# 方法1：直接使用绝对路径
writer = SummaryWriter('G:/postgraduate/anquan/CBF_robu_lujin_plus_kf')


# def sigmoid(x):
#     fai = 1 / (1 + np.e ** (-x))
#     return fai


# 在全局区域或类内部定义可学习的权重参数
num_neurons = 6
# 初始化权重和偏置
np.random.seed(42)
W = np.random.randn(num_neurons, 2) * 0.5  # 形状: (6, 2)
b = np.random.randn(num_neurons, 1) * 0.1  # 形状: (6, 1)

# 原始的sigmoid函数，现在使用外部的W和b
def sigmoid(x, W=W, b=b):
    """
    为2x1输入向量生成Sigmoidal基函数向量。

    参数:
    x : numpy.ndarray - 形状为(2, 1)的输入状态列向量。
    W : numpy.ndarray - 权重矩阵 (6, 2)
    b : numpy.ndarray - 偏置向量 (6, 1)

    返回:
    phi : numpy.ndarray - 形状为(num_neurons, 1)的基函数（激活）向量。
    """
    x = np.asarray(x)
    if x.shape != (2, 1):
        raise ValueError(f"输入x必须是形状为(2,1)的向量，但得到的是 {x.shape}")

    # 计算神经元输入
    z = W @ x + b  # 形状: (6, 1)

    # 应用Sigmoid激活函数
    phi = 1 / (1 + np.exp(-z))*0.000001

    return phi

# 设置速度参数
alpha = np.radians(0)
beta = np.radians(0)
miu_0 = 4 * np.pi * 10 ** -7
frequency = 5  # 频率，单位：Hz

dao = 80  # 自适应增益，调整自适应的快慢
theta_big = 0  # 自适应参数矩阵，对位置理想参数theta_big*的实时估计，初始设置为0
theta_big_pre = 0  # 自适应参数矩阵，对位置理想参数theta_big*的实时估计，初始设置为0
theta_big_pre2 = np.array([[0], [0], [0], [0], [0], [0]])  # 2×1
finally_torque = 0
P = 1 / 20  # Lyapunov方程求解出来的权重矩阵
P_quan= np.array([[1/20], [1/20]])

miu_v = 0.5  # v修正参数，为核心创新。取值在0-0.5

# 磁场幅值
amplitude = 0.0011  # 单位 T
theta_control = np.pi / 1.5  # 固定旋转磁场的旋转角度
r = 30  # 半径 (mm)
# v_max = 2.5    # 恒定速度 (m/s)
v_max = 5  # 恒定速度 (m/s)
omega = v_max / r  # 角速度 (rad/s)
T = 1.5 * np.pi / omega  # 运动周期 (s)

# =====================
# 生成轨迹数据
# =====================
t_start = 0  # 起始时间 (s)
t_end = T  # 结束时间 (s) - 完整周期
dt = 0.035  # 时间步长 (s)
dt_ms = 35

det_L_augumented_standard = 1e-23
last_current_error_robust_weight = 0.95
Ks = np.zeros((4, 1))

obs1_start_position = np.array([5, 9])
obs2_start_position = np.array([-20.0, -10.0])
obs1_end_position = np.array([-36, 9])
obs2_end_position = np.array([6.0, -10.0])
obs_radius = 1.4
theta_pre = np.pi / 2
phi_theta = np.pi / 4
phi_theta_pre = np.pi / 4
optimal_control_loss = 0
theta_zitai_pre=0
theta_zitai=0

def record_tensorboard_data(i_motion, time, current_pos, current_vel, error, dot_error,
                            u_safe, desired_vel_raw, desired_field_strength, theta_applied,
                            cbf_active, current_omega, desired_omega, actual_torque,
                            finally_torque, theta_big, phi_theta, IHx, IHy, IHz,
                            actual1,actual2,
                            optimal_control_loss=0,
                            theta_zitai=0,
                            b_F_values=None,  # 新增参数
                            k_F_values=None):  # 新增参数
    """
    记录TensorBoard数据，包括CBF参数 b_F 和 k_F
    """

    # 1. 记录机器人的速度
    vel_norm = np.linalg.norm(current_vel)
    writer.add_scalar('Robot/Velocity/norm', vel_norm, i_motion)
    writer.add_scalar('Robot/Velocity/x', current_vel[0, 0], i_motion)
    writer.add_scalar('Robot/Velocity/y', current_vel[1, 0], i_motion)

    # 2. 记录角速度
    writer.add_scalar('Robot/AngularVelocity/current', current_omega, i_motion)
    writer.add_scalar('Robot/AngularVelocity/desired', desired_omega, i_motion)
    writer.add_scalar('Robot/AngularVelocity/error', desired_omega - current_omega, i_motion)
    writer.add_scalar('Robot/zitai', theta_zitai, i_motion)

    # 3. 记录位置
    writer.add_scalar('Robot/Position/x', current_pos[0, 0], i_motion)
    writer.add_scalar('Robot/Position/y', current_pos[1, 0], i_motion)

    # 4. 记录朝向角theta
    writer.add_scalar('Robot/Orientation/theta', magnetic_model.theta, i_motion)
    writer.add_scalar('Robot/Orientation/theta_applied', theta_applied, i_motion)

    # 5. 记录速度误差
    vel_error_norm = np.linalg.norm(dot_error)
    writer.add_scalar('Error/Velocity/norm', vel_error_norm, i_motion)
    writer.add_scalar('Error/Velocity/x', dot_error[0, 0], i_motion)
    writer.add_scalar('Error/Velocity/y', dot_error[1, 0], i_motion)

    # 6. 记录角速度误差
    omega_error = desired_omega - current_omega
    writer.add_scalar('Error/AngularVelocity', omega_error, i_motion)

    # 7. 记录位置误差
    pos_error_norm = np.linalg.norm(error)
    writer.add_scalar('Error/Position/norm', pos_error_norm, i_motion)
    writer.add_scalar('Error/Position/x', error[0, 0], i_motion)
    writer.add_scalar('Error/Position/y', error[1, 0], i_motion)

    # 8. 记录XYZ三轴电流
    writer.add_scalar('Current/IHx', IHx, i_motion)
    writer.add_scalar('Current/IHy', IHy, i_motion)
    writer.add_scalar('Current/IHz', IHz, i_motion)
    current_norm = np.sqrt(IHx ** 2 + IHy ** 2 + IHz ** 2)
    writer.add_scalar('Current/norm', current_norm, i_motion)

    # # 9. 记录扭矩
    # if isinstance(actual_torque, np.ndarray):
    #     actual_torque_norm = np.linalg.norm(actual_torque)
    # else:
    #     actual_torque_norm = abs(actual_torque)
    writer.add_scalar('Torque/actual1', actual1, i_motion)
    writer.add_scalar('Torque/actual2', actual2, i_motion)
    # writer.add_scalar('Torque/final', finally_torque, i_motion)
    # writer.add_scalar('Torque/actual_norm', actual_torque_norm, i_motion)
    # writer.add_scalar('Torque/final', finally_torque, i_motion)
    #

    # 新增：记录自适应参数theta_big
    if theta_big is not None:
        # 确保theta_big是numpy数组
        if isinstance(theta_big, np.ndarray):
            # 如果是(6,1)形状，展平为(6,)
            if theta_big.shape == (6, 1):
                theta_big_flat = theta_big.flatten()
            elif theta_big.shape == (6,):
                theta_big_flat = theta_big
            else:
                print(f"警告: theta_big形状异常: {theta_big.shape}")
                theta_big_flat = np.zeros(6)
        else:
            # 如果是标量或其他类型，创建数组
            theta_big_flat = np.array([theta_big] * 6)

        # 分别记录每个元素
        for i in range(len(theta_big_flat)):
            writer.add_scalar(f'Adaptive/theta_big_{i}', theta_big_flat[i], i_motion)

        # 记录统计信息
        writer.add_scalar('Adaptive/theta_big_norm', np.linalg.norm(theta_big_flat), i_motion)
        writer.add_scalar('Adaptive/theta_big_mean', np.mean(theta_big_flat), i_motion)
        writer.add_scalar('Adaptive/theta_big_std', np.std(theta_big_flat), i_motion)

        # 批量记录（分组显示）
        theta_dict = {f'element_{i}': theta_big_flat[i] for i in range(len(theta_big_flat))}
        writer.add_scalars('Adaptive/theta_big_components', theta_dict, i_motion)

        # 记录变化率（如果保存了上一个值）
        if hasattr(record_tensorboard_data, 'last_theta_big'):
            change = np.linalg.norm(theta_big_flat - record_tensorboard_data.last_theta_big)
            writer.add_scalar('Adaptive/theta_big_change', change, i_motion)

        # 保存当前值用于下次计算变化
        record_tensorboard_data.last_theta_big = theta_big_flat.copy()

    # 11. 记录CBF求解出来的速度
    safe_vel_norm = np.linalg.norm(u_safe)
    nominal_vel_norm = np.linalg.norm(desired_vel_raw)
    writer.add_scalar('CBF/Velocity/safe_norm', safe_vel_norm, i_motion)
    writer.add_scalar('CBF/Velocity/nominal_norm', nominal_vel_norm, i_motion)
    writer.add_scalar('CBF/Velocity/difference', safe_vel_norm - nominal_vel_norm, i_motion)
    writer.add_scalar('CBF/Velocity/safe_x', u_safe[0, 0], i_motion)
    writer.add_scalar('CBF/Velocity/safe_y', u_safe[1, 0], i_motion)
    writer.add_scalar('CBF/active', int(cbf_active), i_motion)

    # 12. 记录旋转磁场角度与磁矩的角度差
    writer.add_scalar('Field/phi_theta', phi_theta, i_motion)
    writer.add_scalar('Field/strength', desired_field_strength, i_motion)

    # 13. 记录其他重要参数
    writer.add_scalar('Time/step', i_motion, i_motion)
    writer.add_scalar('Time/actual_time', time, i_motion)
    writer.add_scalar('OptimalControl/Cumulative_Loss_J', optimal_control_loss, i_motion)

    # 14. 记录CBF参数 b_F 和 k_F（新增）
    if b_F_values is not None and len(b_F_values) > 0:
        mean_b_F = np.mean(b_F_values)
        max_b_F = np.max(b_F_values)
        min_b_F = np.min(b_F_values)

        writer.add_scalar('CBF/b_F/mean', mean_b_F, i_motion)
        writer.add_scalar('CBF/b_F/max', max_b_F, i_motion)
        writer.add_scalar('CBF/b_F/min', min_b_F, i_motion)
        writer.add_scalar('CBF/b_F/count', len(b_F_values), i_motion)

    if k_F_values is not None and len(k_F_values) > 0:
        mean_k_F = np.mean(k_F_values)
        writer.add_scalar('CBF/k_F/mean', mean_k_F, i_motion)
        writer.add_scalar('CBF/k_F/count', len(k_F_values), i_motion)

        # 记录每个障碍物的k_F值
        for idx, k_F_val in enumerate(k_F_values):
            writer.add_scalar(f'CBF/k_F/obs{idx}', k_F_val, i_motion)


class CBFPathPlanner:
    def __init__(self):
        self.obstacles = []  # 存储障碍物边界点
        self.area_size = 100  # 规划区域大小
        self.step_size = 30.0  # RRT步长
        self.max_iterations = 1000  # 最大迭代次数
        self.goal_sample_rate = 0.1  # 目标采样率
        self.safety_margin = 6  # 增加安全边界距离
        self.load_boundary_points_from_excel()  # 初始化时自动加载边界点

    def load_boundary_points_from_excel(self, filename="boundary_points.xlsx"):
        """从Excel文件加载障碍物边界点"""
        try:
            df = pd.read_excel(filename)

            if df.empty:
                print("Excel文件为空，使用默认障碍物")
                self.set_default_obstacles()
                return

            # 按障碍物ID分组
            obstacle_ids = df['obs_id'].unique()
            self.obstacles = []

            for obs_id in obstacle_ids:
                # 获取该障碍物的所有边界点
                obs_data = df[df['obs_id'] == obs_id]

                if obs_data.empty:
                    continue

                # 提取所有边界点的相对坐标
                boundary_points = []
                for index, row in obs_data.iterrows():
                    boundary_points.append(np.array([row['rel_X'], row['rel_Y']]))

                obstacle_info = {
                    'id': int(obs_id),
                    'boundary_points': boundary_points,  # 所有边界点
                    'point_count': len(boundary_points)
                }

                self.obstacles.append(obstacle_info)
                print(f"加载障碍物{obs_id}: {len(boundary_points)}个边界点")

            print(f"✅ 从Excel成功加载 {len(self.obstacles)} 个障碍物的边界点")

        except Exception as e:
            print(f"❌ 读取Excel文件失败: {e}")
            self.set_default_obstacles()

    def set_default_obstacles(self):
        """设置默认障碍物（备用）"""
        self.obstacles = [
            {
                'id': 1,
                'boundary_points': [
                    np.array([15, 15]),
                    np.array([25, 15]),
                    np.array([25, 25]),
                    np.array([15, 25])
                ]
            }
        ]
        print("使用默认障碍物设置")

    def reload_boundary_points(self, filename="boundary_points.xlsx"):
        """重新加载边界点"""
        print("=== 重新加载边界点数据 ===")
        self.load_boundary_points_from_excel(filename)

    def is_collision_free(self, point):
        """检查点是否与障碍物碰撞（增加安全距离）"""
        for obstacle in self.obstacles:
            for boundary_point in obstacle['boundary_points']:
                # 计算点到边界点的距离
                dist = np.linalg.norm(point - boundary_point)
                if dist < self.safety_margin:  # 使用安全边界距离
                    return False
        return True

    def get_nearest_node(self, nodes, random_point):
        """在树中找到最近的节点"""
        min_dist = float('inf')
        nearest_node = None

        for node in nodes:
            dist = np.linalg.norm(node - random_point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def steer(self, from_node, to_point):
        """从from_node向to_point方向生长一步"""
        direction = to_point - from_node
        distance = np.linalg.norm(direction)

        if distance > self.step_size:
            direction = direction / distance * self.step_size

        new_point = from_node + direction

        # 检查路径是否碰撞（使用安全距离）
        if self.is_collision_free(new_point):
            return new_point
        else:
            return None

    def plan_path(self, start_pos, goal_pos):
        """RRT路径规划"""
        print(f"开始RRT路径规划: {start_pos} -> {goal_pos}")

        # 初始化树
        tree = [start_pos.copy()]
        parent = {tuple(start_pos): None}

        for iteration in range(self.max_iterations):
            # 随机采样（有一定概率采样目标点）
            if np.random.random() < self.goal_sample_rate:
                random_point = goal_pos.copy()
            else:
                random_point = np.array([
                    np.random.uniform(-self.area_size / 2, self.area_size / 2),
                    np.random.uniform(-self.area_size / 2, self.area_size / 2)
                ])

            # 找到最近的节点
            nearest_node = self.get_nearest_node(tree, random_point)

            # 向随机点方向生长
            new_node = self.steer(nearest_node, random_point)

            if new_node is not None:
                tree.append(new_node)
                parent[tuple(new_node)] = tuple(nearest_node)

                # 检查是否到达目标
                if np.linalg.norm(new_node - goal_pos) < 2.0:
                    print(f"RRT规划成功，迭代次数: {iteration}")
                    return self.extract_path(parent, new_node, start_pos)

        print("RRT规划失败，达到最大迭代次数")
        return None

    def extract_path(self, parent, goal_node, start_pos):
        """从目标节点回溯提取路径"""
        path = [goal_node.copy()]
        current = tuple(goal_node)

        while current != tuple(start_pos):
            current = parent[current]
            path.append(np.array(current))

        path.reverse()
        print(f"提取路径点数: {len(path)}")
        return np.array(path)

    def plan_path_with_smoothing(self, start_pos, goal_pos):
        """带平滑的RRT路径规划"""
        # 先进行RRT规划
        path = self.plan_path(start_pos, goal_pos)

        if path is None:
            return None

        # 路径平滑，同时确保平滑后的路径也远离障碍物
        smoothed_path = self.smooth_path_with_safety(path)
        print(f"平滑后路径点数: {len(smoothed_path)}")
        return smoothed_path

    def smooth_path_with_safety(self, path):
        """带安全性的路径平滑"""
        if len(path) < 3:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            # 每三个点保留一个，减少路径点数量
            if i % 3 == 0:
                # 检查这个点是否安全
                if self.is_collision_free(path[i]):
                    smoothed.append(path[i])
                else:
                    # 如果不安全，跳过这个点
                    continue

        smoothed.append(path[-1])

        # 二次检查所有平滑后的点是否都安全
        final_path = []
        for point in smoothed:
            if self.is_collision_free(point):
                final_path.append(point)
            else:
                # 如果点不安全，尝试在周围寻找安全点
                safe_point = self.find_safe_point_nearby(point)
                if safe_point is not None:
                    final_path.append(safe_point)

        return np.array(final_path)

    def find_safe_point_nearby(self, point, search_radius=2.0):
        """在点周围寻找安全点"""
        for angle in np.linspace(0, 2 * np.pi, 8):  # 8个方向
            for radius in np.linspace(1.0, search_radius, 3):  # 3个距离
                candidate = point + np.array([radius * np.cos(angle), radius * np.sin(angle)])
                if self.is_collision_free(candidate):
                    return candidate
        return None

    def set_safety_margin(self, margin):
        """动态设置安全边界距离"""
        self.safety_margin = margin
        print(f"安全边界距离设置为: {margin}")


# 全局路径规划器实例
path_planner = CBFPathPlanner()


class CBFController:
    def __init__(self):
        self.v_safe_margin = 0.5
        self.obstacles = [
            {'position': obs1_start_position, 'radius': obs_radius, 'end_pos': obs1_end_position},
            {'position': obs2_start_position, 'radius': obs_radius, 'end_pos': obs2_end_position}
        ]
        self.safe_distance = 3.0
        self.gamma = 1.8

        # 改进的参数设置
        self.activation_threshold = 0.8
        self.k_F0 = 1.0
        self.alpha_F_gain = 0.5

        # 新增平滑处理参数
        self.smoothing_factor = 0.3  # 控制输出平滑度 (0-1)
        self.last_safe_control = None
        self.k_F_smoothing = 0.5  # k_F参数平滑系数
        self.last_k_F = self.k_F0

        # 新增：控制共享性质相关参数
        self.safety_k1 = 1.0  # HOCBF参数
        self.safety_k2 = 1.0

        # 存储历史信息
        self.obstacle_history = []

        self._init_obstacle_states()

    def _init_obstacle_states(self):
        """初始化障碍物状态"""
        for obs in self.obstacles:
            obs['active'] = False
            obs['k_F'] = self.k_F0
            obs['hx'] = 0
            obs['h_grad'] = np.zeros(2)
            obs['activation_level'] = 0.0
            obs['last_k_F'] = self.k_F0  # 用于平滑
            obs['L_g_b_F'] = None
            obs['L_g_L_f_b'] = None

    def _smooth_activation(self, distance, influence_distance, activation_distance):
        """平滑的激活函数，避免硬切换"""
        if distance >= influence_distance:
            return 0.0
        elif distance <= activation_distance:
            return 1.0
        else:
            # 在激活距离和影响距离之间平滑过渡
            return 1.0 - (distance - activation_distance) / (influence_distance - activation_distance)

    def compute_feasibility_constraint(self, hx, h_grad, u_limits):
        """计算可行性约束函数 b_F(x) = γ·h(x) - u_l(x) (论文公式18)"""
        u_min, u_max = u_limits

        if h_grad is not None and len(h_grad) > 0:
            # 计算 u_l(x) - 输入约束在安全方向上的最小能力 (论文公式15)
            u_l = np.sum([min(-h_grad_i * u_min_i, -h_grad_i * u_max_i)
                          for h_grad_i, u_min_i, u_max_i in zip(h_grad, u_min, u_max)])
        else:
            u_l = 0

        # 简化版：b_F(x) = γ·h(x) - u_l(x)
        # 论文完整版：b_F(x) = L_f^m b(x) + S(b(x)) + α_m(ψ_{m-1}(x)) - u_l(x)
        b_F = self.gamma * hx - u_l

        return b_F

    def compute_control_sharing_constraints(self, k_F, b_F, L_g_b_F, L_g_L_f_b, c_value):
        """计算控制共享性质约束 (论文定理1的关键)"""
        constraints = []

        # 检查每个控制维度
        for k in range(len(L_g_b_F)):
            if abs(L_g_b_F[k]) > 1e-8 and abs(L_g_L_f_b[k]) > 1e-8:
                # 计算符号 (论文中的sign函数)
                sgn_L_g_b_F = 1 if L_g_b_F[k] > 0 else -1
                sgn_L_g_L_f_b = 1 if L_g_L_f_b[k] > 0 else -1

                # 论文公式(22)的第二项约束
                if sgn_L_g_b_F * sgn_L_g_L_f_b == -1:
                    # 方向相反的情况，需要额外约束保证兼容性
                    compatibility_term = (
                            k_F * self.alpha_F_gain * max(b_F, 0) +
                            self.gamma * b_F -
                            (L_g_b_F[k] / L_g_L_f_b[k]) * c_value
                    )

                    # 乘以 (1 - sign(L_g b_F)sign(L_g L_f b))，这里为2
                    constraint = 2 * compatibility_term >= 0
                    constraints.append(constraint)

        return constraints

    def cbf_constraints(self, current_pos, current_vel, control_input, current_time):
        """生成CBF约束"""
        constraints = []
        current_pos = current_pos.flatten()

        progress = min(current_time / T, 1.0)

        for i, obs in enumerate(self.obstacles):
            obs_pos = obs['position'] + progress * (obs['end_pos'] - obs['position'])
            to_obs = current_pos - obs_pos
            distance = np.linalg.norm(to_obs)

            influence_distance = self.safe_distance + obs['radius']
            activation_distance = influence_distance * self.activation_threshold

            # 计算平滑的激活级别
            activation_level = self._smooth_activation(distance, influence_distance, activation_distance)
            obs['activation_level'] = activation_level

            if activation_level > 0.01:
                obs['active'] = True

                if distance > 1e-8:
                    h_grad = to_obs / distance
                else:
                    h_grad = np.array([1.0, 0.0])

                # 计算安全函数值
                effective_safe_distance = self.safe_distance * (1.0 + 0.5 * (1 - activation_level))
                hx = distance - (effective_safe_distance + obs['radius'])

                # 调整gamma参数
                effective_gamma = self.gamma * activation_level

                # CBF约束 (简化的一阶版本)
                # 论文中的完整版：L_f^m b + L_g L_f^{m-1} b u + S(b) + α_m(ψ_{m-1}) ≥ 0
                cbf_constraint = h_grad @ control_input >= -effective_gamma * hx
                constraints.append(cbf_constraint)

                # 保存必要信息用于可行性约束
                obs['hx'] = hx
                obs['h_grad'] = h_grad
                obs['current_pos'] = obs_pos
                obs['distance'] = distance
                obs['effective_gamma'] = effective_gamma

                # 简化的李导数项 (实际应用中需要根据具体动力学计算)
                obs['L_g_b_F'] = h_grad  # 简化假设
                obs['L_g_L_f_b'] = h_grad  # 简化假设
                obs['c_value'] = effective_gamma * hx  # 对应论文中的L_f^m b + S(b) + α_m(ψ)

            else:
                obs['active'] = False

        return constraints

    def solve_inner_qp_with_sharing(self, b_F, h_grad, u_limits, activation_level, obstacle_idx):
        """改进的内层QP：包含控制共享性质约束"""
        try:
            k_F = cp.Variable(1)
            last_k_F = self.obstacles[obstacle_idx]['last_k_F']

            # 成本函数：希望k_F接近上一次的值（平滑性）
            cost = cp.sum_squares(k_F - last_k_F)

            # 约束条件
            constraints = [k_F >= 0.01, k_F <= 10.0]  # 合理的范围

            u_min, u_max = u_limits
            if h_grad is not None and len(h_grad) > 0:
                # 1. 输入兼容性约束 (论文公式22最后一个约束)
                min_control_effort = np.sum([min(-h_grad_i * u_min_i, -h_grad_i * u_max_i)
                                             for h_grad_i, u_min_i, u_max_i in zip(h_grad, u_min, u_max)])

                adjusted_b_F = b_F * activation_level
                input_compatibility = k_F * self.alpha_F_gain * max(adjusted_b_F, 0.001) - min_control_effort >= 0
                constraints.append(input_compatibility)

                # 2. 控制共享性质约束
                # 获取之前保存的李导数信息
                L_g_b_F = self.obstacles[obstacle_idx].get('L_g_b_F', h_grad)
                L_g_L_f_b = self.obstacles[obstacle_idx].get('L_g_L_f_b', h_grad)
                c_value = self.obstacles[obstacle_idx].get('c_value', 0)

                sharing_constraints = self.compute_control_sharing_constraints(
                    k_F, b_F, L_g_b_F, L_g_L_f_b, c_value
                )
                constraints.extend(sharing_constraints)

            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=100)

            if prob.status == cp.OPTIMAL:
                k_F_value = float(k_F.value[0])

                # 平滑处理
                smoothed_k_F = (self.k_F_smoothing * last_k_F +
                                (1 - self.k_F_smoothing) * k_F_value)

                # 更新
                self.obstacles[obstacle_idx]['last_k_F'] = smoothed_k_F

                # 调试信息
                if len(sharing_constraints) > 0:
                    print(f"  障碍物{obstacle_idx}: k_F={smoothed_k_F:.3f}, 添加{len(sharing_constraints)}个共享约束")

                return smoothed_k_F

        except Exception as e:
            print(f"  障碍物{obstacle_idx}: 内层QP异常: {str(e)[:50]}")

        return self.obstacles[obstacle_idx]['last_k_F']

    def smooth_control_output(self, new_control, nominal_control):
        """平滑控制输出，避免突变"""
        if self.last_safe_control is None:
            self.last_safe_control = new_control
            return new_control

        smoothed_control = (self.smoothing_factor * self.last_safe_control +
                            (1 - self.smoothing_factor) * new_control)

        # 限制最大偏离
        deviation = np.linalg.norm(smoothed_control - nominal_control)
        max_deviation = np.linalg.norm(nominal_control) * 0.8

        if deviation > max_deviation:
            direction = (smoothed_control - nominal_control) / deviation
            smoothed_control = nominal_control + direction * max_deviation

        self.last_safe_control = smoothed_control
        return smoothed_control

    def optimize_control(self, current_pos, current_vel, nominal_control, current_time):
        """CBF优化求解安全控制量 - 集成控制共享性质"""
        self._init_obstacle_states()

        u_opt = cp.Variable(2)

        # 成本函数
        control_diff = u_opt - nominal_control.flatten()
        cost = cp.sum_squares(control_diff)

        if self.last_safe_control is not None:
            rate_penalty = 0.1 * cp.sum_squares(u_opt - self.last_safe_control.flatten())
            cost += rate_penalty

        # 1. 生成CBF约束
        cbf_constraints = self.cbf_constraints(current_pos, current_vel, u_opt, current_time)

        # 2. 添加可行性约束（包含控制共享性质）
        feasibility_constraints = []
        u_limits = ([-200, -200], [200, 200])

        active_obstacles = []
        for i, obs in enumerate(self.obstacles):
            if obs['active'] and obs['activation_level'] > 0.1:
                active_obstacles.append(i)

                # 计算可行性函数
                b_F = self.compute_feasibility_constraint(obs['hx'], obs['h_grad'], u_limits)

                # 关键改进：使用包含控制共享性质的内层QP
                k_F_optimal = self.solve_inner_qp_with_sharing(
                    b_F, obs['h_grad'], u_limits,
                    obs['activation_level'], i
                )

                obs['k_F'] = k_F_optimal

                if obs['h_grad'] is not None:
                    # 添加可行性约束
                    adjusted_b_F = b_F * obs['activation_level']
                    feasibility_constraint = (
                            obs['h_grad'] @ u_opt +
                            k_F_optimal * self.alpha_F_gain * max(adjusted_b_F, 0) >= 0
                    )
                    feasibility_constraints.append(feasibility_constraint)

        # 3. 控制输入约束
        control_constraints = [
            cp.norm(u_opt, 2) <= 300,
            u_opt[0] >= -200, u_opt[1] >= -200,
            u_opt[0] <= 200, u_opt[1] <= 200
        ]

        # 4. 合并所有约束
        all_constraints = cbf_constraints + feasibility_constraints + control_constraints

        print(f"时间{current_time:.2f}s: 激活障碍物{active_obstacles}, "
              f"约束数: CBF={len(cbf_constraints)}, 可行性={len(feasibility_constraints)}")

        prob = cp.Problem(cp.Minimize(cost), all_constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status == cp.OPTIMAL:
                raw_control = u_opt.value.reshape(2, 1)

                # 平滑处理
                safe_control = self.smooth_control_output(raw_control, nominal_control)

                # 检查CBF是否起作用
                cbf_active = not np.allclose(safe_control, nominal_control, atol=0.5)

                if active_obstacles:
                    k_F_values = [self.obstacles[i]['k_F'] for i in active_obstacles]
                    status = "CBF激活" if cbf_active else "CBF监控"
                    print(f"{status}: [{safe_control[0, 0]:.2f}, {safe_control[1, 0]:.2f}], "
                          f"障碍物{active_obstacles}, k_F={k_F_values}")
                else:
                    print(f"名义控制: [{safe_control[0, 0]:.2f}, {safe_control[1, 0]:.2f}]")

                return safe_control
            else:
                print(f"CBF优化失败，使用名义控制")
                return self.smooth_control_output(nominal_control, nominal_control)

        except Exception as e:
            print(f"CBF求解异常: {e}，使用名义控制")
            return self.smooth_control_output(nominal_control, nominal_control)


cbf_controller = CBFController()
magnetic_model = model.magnetic_motion_model(dt=dt, position=[[1], [8]], vel=[[0.0001], [0.0001]], theta=np.pi / 2,
                                             current_limit=10)


def generate_follow_path(target_pos):
    global magnetic_model

    # 设置起点和终点
    start_pos = magnetic_model.return_position().reshape(-1)  # 机器人起始位置
    goal_pos = target_pos  # 目标位置

    print("开始路径规划...")
    planned_path = path_planner.plan_path(start_pos, goal_pos)
    print(f"规划路径点数: {len(planned_path)}")

    # 将规划路径转换为时间序列
    magnetic_model.ref_time = np.arange(t_start, t_end + dt, dt)
    N = len(magnetic_model.ref_time)

    if len(planned_path) > N:
        # 如果规划路径比时间序列长，进行采样
        indices = np.linspace(0, len(planned_path) - 1, N).astype(int)
        path_xy = planned_path[indices]
    else:
        # 如果规划路径短，进行插值
        from scipy import interpolate
        t_planned = np.linspace(0, 1, len(planned_path))
        t_desired = np.linspace(0, 1, N)

        spline_x = interpolate.CubicSpline(t_planned, planned_path[:, 0])
        spline_y = interpolate.CubicSpline(t_planned, planned_path[:, 1])

        path_xy = np.column_stack((spline_x(t_desired), spline_y(t_desired)))

    # 设置参考轨迹
    magnetic_model.ref_x = path_xy[:, 0]
    magnetic_model.ref_y = path_xy[:, 1]

    # 计算参考速度（数值微分）
    magnetic_model.ref_vx = np.gradient(magnetic_model.ref_x, magnetic_model.ref_time)
    magnetic_model.ref_vy = np.gradient(magnetic_model.ref_y, magnetic_model.ref_time)

    # 计算参考加速度
    magnetic_model.ref_ax = np.gradient(magnetic_model.ref_vx, magnetic_model.ref_time)
    magnetic_model.ref_ay = np.gradient(magnetic_model.ref_vy, magnetic_model.ref_time)

    # 动态障碍物轨迹
    tn = magnetic_model.ref_time / T
    magnetic_model.obs1_x = obs1_start_position[0] + (obs1_end_position[0] - obs1_start_position[0]) * tn
    magnetic_model.obs1_y = obs1_start_position[1] + (obs1_end_position[1] - obs1_start_position[1]) * tn
    magnetic_model.obs2_x = obs2_start_position[0] + (obs2_end_position[0] - obs2_start_position[0]) * tn
    magnetic_model.obs2_y = obs2_start_position[1] + (obs2_end_position[1] - obs2_start_position[1]) * tn

    # 组合路径数据
    magnetic_model.ref_path = np.column_stack((magnetic_model.ref_x, magnetic_model.ref_y))
    magnetic_model.ref_velpath = np.column_stack((magnetic_model.ref_vx, magnetic_model.ref_vy))
    magnetic_model.ref_accelpath = np.column_stack((magnetic_model.ref_ax, magnetic_model.ref_ay))

    print("路径规划完成!")
    print(f"起点: {start_pos}, 终点: {goal_pos}")
    print(f"参考路径点数: {len(magnetic_model.ref_path)}")


def return_current_position():
    global magnetic_model
    return magnetic_model.return_position()


def update_status(current_positon):
    global magnetic_model
    magnetic_model.update_status(current_positon)


# clf_initialization
K1 = np.array([[1, 0, 1, 0],
               [0, 1, 0, 1]])
Q = np.eye(4)
epsilon = 1
c3 = 1

import cv2
import numpy as np

kf = cv2.KalmanFilter(4, 2)  # 4 维状态（x, y, vx, vy），2 维观测（x, y）
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1


def calculate_angle(A, B):
    delta_y = B[1] - A[1]
    delta_x = B[0] - A[0]
    # 这边计算角度的没懂，不过应该是微分法，里头应该存了上一时刻的位置和本时刻位置
    angle = np.arctan2(delta_y, delta_x)  # 计算角度（弧度）
    theta = angle if angle >= 0 else angle + 2 * np.pi
    # 保证是正数
    return theta


cnt = 0


def adjust_angle():
    # 小作弊的用法，用于直接设定初始位移角度
    global cnt
    magnetic_model.theta = calculate_angle(magnetic_model.return_position().reshape(-1),
                                           magnetic_model.ref_path[0]) - 1 / 2 * np.pi
    # magnetic_model.vel = np.array([[0.001*np.cos(magnetic_model.theta)], [0.001*np.sin(magnetic_model.theta)]])
    u_x, u_y, u_z = np.sin(alpha), -np.cos(alpha), 0  # 假设 x 方向单位向量
    v_x, v_y, v_z = np.cos(alpha) * np.cos(magnetic_model.theta), np.sin(alpha) * np.cos(magnetic_model.theta), -np.sin(
        magnetic_model.theta)  # 转向
    # cHx = amplitude * (u_x * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_x * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    # cHz = amplitude * (u_y * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_y * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    # cHy = amplitude * (u_z * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_z * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHx = amplitude * (u_x * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_x * np.sin(
        2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHz = amplitude * (u_y * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_y * np.sin(
        2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHy = amplitude * (u_z * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_z * np.sin(
        2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    # 生成对应的电流
    IHx = cHx / ((4 / 5) ** (3 / 2) * miu_0 * 297 / 0.236)
    IHz = cHz / ((4 / 5) ** (3 / 2) * miu_0 * 202 / 0.162)
    IHy = cHy / ((4 / 5) ** (3 / 2) * miu_0 * 129 / 0.1)

    # cnt += 1
    return IHx, IHy, IHz


def calculate_magnetic_moment_orientation(rotation_angle, robot_orientation):
    """
    根据旋转角度和机器人朝向计算磁矩朝向角

    参数:
        rotation_angle: 磁场旋转角度 (theta_control * time)
        robot_orientation: 机器人朝向角 (magnetic_model.theta)

    返回:
        magnetic_moment_angle: 磁矩朝向角 (弧度)
    """
    # 磁矩方向相对于机器人本体系的角度
    # 假设磁矩在机器人坐标系中的初始方向
    magnetic_moment_relative = rotation_angle

    # 转换到全局坐标系
    magnetic_moment_angle = (robot_orientation + magnetic_moment_relative) % (2 * np.pi)

    return magnetic_moment_angle


part2 = 0


def motion_control(i_motion, dt):
    global magnetic_model, part2, beita, cost_state, L_state, exceed_flag_state, Ks, last_current_error_robust
    global theta_pre, phi_theta, theta_big, phi_theta_pre
    # 新增：最优控制修正相关变量
    global optimal_control_loss ,b ,W,theta_zitai,theta_zitai_pre

    if i_motion < len(magnetic_model.ref_time):
        time, xd, yd, vxd, vyd, axd, ayd = magnetic_model.ref_time[i_motion], magnetic_model.ref_x[i_motion], \
        magnetic_model.ref_y[i_motion], magnetic_model.ref_vx[i_motion], magnetic_model.ref_vy[i_motion], \
        magnetic_model.ref_ax[i_motion], magnetic_model.ref_ay[i_motion]
        beita = np.zeros((4, 1))
        if time < 0.07:
            last_current_error_robust = np.zeros((4, 1))

        magnetic_model.path_log.append(magnetic_model.position)
        if i_motion == 0:
            kf.statePost = np.array(
                [[magnetic_model.position[0, 0]], [magnetic_model.position[1, 0]], [magnetic_model.vel[0, 0]],
                 [magnetic_model.vel[1, 0]]], dtype=np.float32)
        else:
            kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                            [0, 1, 0, dt],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
            # 预测
            prediction = kf.predict()
            # 更新
            kf.correct(magnetic_model.return_position().reshape(-1).astype(np.float32))
            # 获取估计的 (x, y) 和速度 (vx, vy)
            estimated_x, estimated_y = kf.statePost[0, 0], kf.statePost[1, 0]
            estimated_vx, estimated_vy = kf.statePost[2, 0], kf.statePost[3, 0]
            magnetic_model.vel = np.array([[estimated_vx], [estimated_vy]])
            magnetic_model.theta = math.atan2(estimated_vy, estimated_vx)
            if magnetic_model.theta < 0:
                magnetic_model.theta += 2 * np.pi

        magnetic_model.error = np.array([[xd], [yd]]) - np.array(magnetic_model.return_position())
        error = magnetic_model.error
        dot_error = np.array([[vxd], [vyd]]) - np.array(magnetic_model.return_vel())
        current_error = np.array([error, dot_error])
        current_error = current_error.reshape(-1, 1)

        magnetic_model.error_log.append(current_error)

        ############################################################################################################################################################
        # 控制器修改只修改######内的内容

        # 2. PD控制器计算期望速度
        Kp = 10.0  # 位置增益
        Kd = 0.5  # 速度增益

        desired_vel_raw = Kp * error + Kd * dot_error

        desired_vel = np.linalg.norm(desired_vel_raw)
        desired_omega = desired_vel / 2
        I = 1e-4  # 转动惯量 [kg·mm²]
        current_omega = np.linalg.norm(magnetic_model.vel) / 2
        # print("现在的omega：°",current_omega*57.3)

        # CBF控制
        u_safe = cbf_controller.optimize_control(np.array(magnetic_model.return_position()),
                                                 np.array([[vxd], [vyd]]), desired_vel_raw, time)
        print("名义控制：", desired_vel_raw)
        print("CBF计算后的速度", u_safe)

        # === 新增：提取 b_F 和 k_F 值 ===
        b_F_values = []
        k_F_values = []

        for obs in cbf_controller.obstacles:
            if obs.get('active', False):
                # 获取 b_F 值（从障碍物信息中提取）
                if 'hx' in obs and 'h_grad' in obs:
                    u_limits = ([-200, -200], [200, 200])
                    b_F = cbf_controller.compute_feasibility_constraint(
                        obs['hx'], obs['h_grad'], u_limits
                    )
                    b_F_values.append(b_F)

                # 获取 k_F 值
                if 'k_F' in obs:
                    k_F_values.append(obs['k_F'])

        # 检查CBF是否起作用（安全速度与名义控制是否不同）
        cbf_active = not np.allclose(u_safe, desired_vel_raw, atol=0.01)

        if cbf_active:
            # CBF起作用：使用安全速度的方向
            print("CBF避障激活！")

            # 计算安全速度的方向
            safe_vel_direction = np.arctan2(u_safe[1, 0], u_safe[0, 0])
            if safe_vel_direction < 0:
                safe_vel_direction += 2 * np.pi

            # 更新磁体朝向为安全速度方向
            magnetic_model.theta = safe_vel_direction
            theta_applied = magnetic_model.theta - 1 / 2 * np.pi

            print(f"CBF避障方向: {theta_applied * 57.3:.1f}°")

        else:
            # CBF未起作用：使用原来的轨迹跟踪方向
            print("CBF未激活，使用轨迹跟踪")
            magnetic_model.theta = calculate_angle(magnetic_model.return_position().reshape(-1),
                                                   magnetic_model.ref_path[i_motion])
            theta_applied = magnetic_model.theta - 1 / 2 * np.pi
            print(f"CBF避障方向: {theta_applied * 57.3:.1f}°")

        # 使用安全速度计算后续参数
        desired_vel = np.linalg.norm(u_safe)  # 使用安全速度的大小
        desired_omega = desired_vel / 2

        # T_inv=np.array([[0.5/0.02, 0.5/(0.02*0.02)],
        #       [0.5/0.02, -0.5/(0.02*0.02)]])
        # 逆矩阵中的角速度对应的元素
        T_inv11= 0.25
        T_inv12= 0.5
        T_inv21 = 0.25
        T_inv22 = -0.5

        # 1.22e-9是溶液阻力系数
        if i_motion > 0:
            vel_mozhi= np.linalg.norm(magnetic_model.vel)
            alpha_omega = (desired_omega - current_omega) / dt  # 角加速度
            alpha_velocity=(desired_vel-vel_mozhi)/dt
            state_rob = np.array([[vel_mozhi], [vel_mozhi/2]])#列向量
            error_robust = np.array([[desired_vel-vel_mozhi],
                                     [desired_omega - current_omega]])#列向量
            # ==================== 最优控制修正核心算法 ====================
            # 根据论文公式(10)计算最优控制修正损失函数 J
            # Q_matrix = np.eye(1)  # Q 是单位矩阵
            # delta = 1.0  # Δ = 1
            # # 当前时刻的最优控制修正损失（瞬时值）
            # current_optimal_loss = 0.5 * (error_robust - delta) * (error_robust - delta)
            # # 累积最优控制修正损失（近似积分）
            # optimal_control_loss = current_optimal_loss


            theta_big_change = -dao * sigmoid(state_rob) @ (
                    error_robust.reshape(1, -1) @ P_quan * (1 / 1e-4) -
                    miu_v * sigmoid(state_rob).T @ theta_big_pre2 * (1 / 1e-4) * P
            )
            theta_big = theta_big_pre2 + theta_big_change * dt
            W=theta_big
            uad = theta_big * sigmoid(state_rob)
            # print("计算出的uad",uad)


            #des_torque=(50 * alpha_velocity + 1.22e-9 * desired_vel , I * alpha_omega + 1.22e-9 * desired_omega)
            desired_torque1 = 0.005 * alpha_velocity + 1.77e-9 * desired_vel
            desired_torque2 = I * alpha_omega + 1.22e-9 * desired_omega
            print("计算出的处理前扭矩desired_torque1", desired_torque1)
            print("计算出的处理前扭矩desired_torque2", desired_torque2)

            actual1= (T_inv11 * desired_torque1 +  T_inv12 * desired_torque2*100)
            actual2= abs((T_inv21 * desired_torque1 + T_inv22 * desired_torque2*100))


            print("计算出的处理后扭矩actual1", actual1)
            print("计算出的处理后扭矩actual2", actual2)
            actual_torque= np.array([[actual1] , [actual2]])  # 列向量

            # actual_torque = T_inv21 * desired_torque1 +  T_inv22 * desired_torque2
            finally_torque=actual_torque
            # finally_torque = min(abs(actual_torque - uad), 1500)
            print("估计后的扭矩：", finally_torque)
        else:
            actual_torque = np.array([[0.0], [0.0]])
            actual1=0
            actual2=0
            finally_torque = 0

        desired_field_strength_hajimi = np.linalg.norm(finally_torque) / (
                    magnetic_model.Magnetic * np.sin(phi_theta_pre))
        # print("计算出的处理前扭矩", desired_torque)
        desired_field_strength = desired_field_strength_hajimi
        print("计算出的actual_torque", actual_torque)
        print("计算出的总的磁场幅值", desired_field_strength)
        # desired_field_strength = np.clip(desired_field_strength, 0.0011, 0.005)
        desired_field_strength = np.clip(desired_field_strength, 0.001, 0.006)
        print("实际的磁场幅值", desired_field_strength)

        # 轨迹跟踪：
        # original controller
        sliding = dot_error + 0.5 * error
        part2 = part2 + 0.1 * np.sign(sliding) * dt
        input = magnetic_model.mass * np.array(
            [[axd], [ayd]]) + magnetic_model.resistance * magnetic_model.vel + 1 * sliding + 1 * np.abs(sliding) ** (
                        1 / 2) * np.sign(sliding) + part2

        print("input:", input)
        ############################################################################################################################################################

        # 这边有问题
        # magnetic_model.input_log.append([input])  # 记录每一时刻的理论输入
        magnetic_model.dynamic_position(input, dt)
        # 后面的包括了史密斯预估器，将F结算为角速度再到磁场旋转速度

        u_ff = magnetic_model.smith_com(input, dt, np.array([[xd], [yd]]), np.array([[vxd], [vyd]]))
        # 其实史密斯预估器这边也不建议更改，还是保留
        input = input + u_ff
        # magnetic_model.dynamic_position(input, dt)
        # 这边有问题
        magnetic_model.input_smith_log.append(input)  # 记录每一时刻的加上smith补偿的理论输入

        # 这边是动力学部分，暂时不考虑更改

        omega_vel = np.linalg.norm((magnetic_model.vel + magnetic_model.accel * dt))
        # omega应该是指磁体要滚得多块，这边对应磁体边长上一点的线速度·
        # (np.linalg.norm)用来求取二范数，这边由于vel里头是velx和vely，所以取二范数
        omega_vel = omega_vel / (2 * np.pi * 2)
        theta_zitai = theta_zitai_pre + omega_vel * dt
        # 使用if语句将姿态角限制在 [0, 2π] 范围内
        if theta_zitai < 0:
            theta_zitai = theta_zitai + 2 * np.pi
        if theta_zitai >= 2 * np.pi:
            theta_zitai = theta_zitai - 2 * np.pi
        # 将线速度转化为角速度，除以2pi*r，这边的r应该是2mm的意思，在更换磁体的时候需要修改
        # print("orignal omgea: ", omega_vel)
        omega_vel = np.clip(omega_vel, 0, 0.4)

        u_x, u_y, u_z = np.sin(alpha), -np.cos(alpha), 0  # 假设 x 方向单位向量
        v_x, v_y, v_z = np.cos(alpha) * np.cos(theta_applied), np.sin(alpha) * np.cos(theta_applied), -np.sin(
            theta_applied)  # 转向
        print("theta情况", theta_applied)

        cHx = desired_field_strength * (u_x * np.cos(theta_control * magnetic_model.ref_time[i_motion]) - v_x * np.sin(
            theta_control * magnetic_model.ref_time[i_motion]))
        cHz = desired_field_strength * (u_y * np.cos(theta_control * magnetic_model.ref_time[i_motion]) - v_y * np.sin(
            theta_control * magnetic_model.ref_time[i_motion]))
        cHy = desired_field_strength * (u_z * np.cos(theta_control * magnetic_model.ref_time[i_motion]) - v_z * np.sin(
            theta_control * magnetic_model.ref_time[i_motion]))

        # field_angle = (theta_control * magnetic_model.ref_time[i_motion]) % (2 * np.pi)
        # 计算当前磁矩朝向角
        rotation_angle = (theta_control * magnetic_model.ref_time[i_motion]) % (2 * np.pi)
        ciju_angle = calculate_magnetic_moment_orientation(rotation_angle, magnetic_model.theta)
        phi_theta = min(abs(rotation_angle - ciju_angle), 2 * np.pi - abs(rotation_angle - ciju_angle))

        distance_to_target = np.linalg.norm(magnetic_model.return_position().reshape(-1) - magnetic_model.ref_path[-1])
        if distance_to_target < 3.0:
            IHx = 0
            IHy = 0
            IHz = 0
        else:
            pass

        # 生成对应的电流
        IHx = cHx / ((4 / 5) ** (3 / 2) * miu_0 * 297 / 0.236)
        IHz = -cHz / ((4 / 5) ** (3 / 2) * miu_0 * 202 / 0.162)
        IHy = -cHy / ((4 / 5) ** (3 / 2) * miu_0 * 129 / 0.1)

        magnetic_model.current_log.append([IHx, IHy, IHz])

        # 在return之前添加记录函数调用
        record_tensorboard_data(
            i_motion=i_motion,
            time=time,
            current_pos=magnetic_model.return_position(),
            current_vel=magnetic_model.return_vel(),
            error=error,
            dot_error=dot_error,
            u_safe=u_safe,
            desired_vel_raw=desired_vel_raw,
            desired_field_strength=desired_field_strength_hajimi,
            theta_applied=theta_applied,
            cbf_active=cbf_active,
            current_omega=current_omega,
            desired_omega=desired_omega,
            actual_torque=actual_torque,
            finally_torque=finally_torque,
            theta_big=theta_big,
            phi_theta=phi_theta,
            IHx=IHx,
            IHy=IHy,
            IHz=IHz,
            actual1=actual1,
            actual2=actual2,
            optimal_control_loss=optimal_control_loss,
            b_F_values=b_F_values,  # 新增：传递b_F值
            k_F_values=k_F_values,  # 新增：传递k_F值
            theta_zitai=theta_zitai
        )

        theta_pre = theta_applied
        phi_theta_pre = phi_theta
        theta_zitai_pre=theta_zitai


    else:
        IHx = 0
        IHy = 0
        IHz = 0
    return IHx, IHy, IHz


def motion_log():
    # 将列表打包成一个字典或列表
    data = {
        "time": magnetic_model.ref_time,
        "ref_path": magnetic_model.ref_path,
        "path": magnetic_model.path_log,
        "error": magnetic_model.error_log,
        "input": magnetic_model.input_log,
        "input_smith": magnetic_model.input_smith_log,
        "valid_input": magnetic_model.delay_input_log,
        "current": magnetic_model.current_log,
        "L_state": magnetic_model.L_state_log,
        "cost_state": magnetic_model.cost_state_log,
        "exceed_flag_state": magnetic_model.exceed_flag_state_log,
        "current_error_robust_state": magnetic_model.current_error_robust_log,
        "det_L_augumented_state": magnetic_model.det_L_augumented_log,
        "Ks_state": magnetic_model.Ks_log
        # "theta": magnetic_model.theta_log
    }

    # 保存到一个 pkl 文件
    with open("ffstc_data.pkl", "wb") as f:
        pickle.dump(data, f)
