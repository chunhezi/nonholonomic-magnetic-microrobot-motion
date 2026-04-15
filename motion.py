import numpy as np
import pickle
import model
import scipy.sparse as sparse
# import osqp
import scipy.linalg as sl
import math
import cvxpy as cp
from simulation import theta

def sigmoid(x):
    fai=1/(1+np.e**(-x))
    return  fai

# 设置速度参数
alpha = np.radians(0)
beta = np.radians(0)
miu_0 = 4 * np.pi * 10 ** -7
frequency = 5  # 频率，单位：Hz

dao=80    #自适应增益，调整自适应的快慢
theta_big=0 #自适应参数矩阵，对位置理想参数theta_big*的实时估计，初始设置为0
theta_big_pre=0 #自适应参数矩阵，对位置理想参数theta_big*的实时估计，初始设置为0
finally_torque=0
P=1/20      #Lyapunov方程求解出来的权重矩阵
miu_v=0.5 #v修正参数，为核心创新。取值在0-0.5

# 磁场幅值
amplitude = 0.0011 # 单位 T
theta_control=np.pi/2   #固定旋转磁场的旋转角度
r = 30        # 半径 (mm)
# v_max = 2.5    # 恒定速度 (m/s)
v_max = 5    # 恒定速度 (m/s)
omega = v_max / r  # 角速度 (rad/s)
T = 3 * np.pi / omega  # 运动周期 (s)

# =====================
# 生成轨迹数据
# =====================
t_start = 0         # 起始时间 (s)
t_end = T           # 结束时间 (s) - 完整周期
dt = 0.035           # 时间步长 (s)
dt_ms = 35

det_L_augumented_standard = 1e-23
last_current_error_robust_weight = 0.95
Ks = np.zeros((4, 1))

obs1_start_position=np.array([ -6,r * np.sin(np.pi/2)-12])
obs2_start_position=np.array([-20.0, -8.0])
obs1_end_position=np.array([ 10,r * np.sin(np.pi/2)-12])
obs2_end_position=np.array([16.0, 8.0])
obs_radius=3
goals_pos=np.array([-130, -50])

class CBFPathPlanner:
    def __init__(self):
        self.obstacles = [
            {'position': obs1_start_position, 'radius': obs_radius},
            {'position': obs2_start_position, 'radius': obs_radius}
        ]
        self.safe_distance = 3
        self.gamma = 1.8
        self.attraction_gain = 2.0  # 目标吸引力
        self.repulsion_gain = 5.0  # 障碍物排斥力

    def compute_potential_field(self, position, goal):
        """计算人工势场"""
        total_force = np.zeros(2)

        # 目标吸引力
        to_goal = goal - position
        dist_to_goal = np.linalg.norm(to_goal)
        if dist_to_goal > 0:
            attraction_force = self.attraction_gain * to_goal / dist_to_goal
            total_force += attraction_force

        # 障碍物排斥力
        for obs in self.obstacles:
            to_obs = position - obs['position']
            dist_to_obs = np.linalg.norm(to_obs)
            safe_dist = self.safe_distance * obs['radius']

            if dist_to_obs < safe_dist:
                if dist_to_obs > 0.1:  # 避免除零
                    repulsion_force = self.repulsion_gain * (1 / dist_to_obs - 1 / safe_dist) * (
                                1 / dist_to_obs ** 2) * (to_obs / dist_to_obs)
                    total_force += repulsion_force

        return total_force

    def plan_path(self, start_pos, goal_pos, num_points=100):
        """基于势场法规划路径"""
        path = [start_pos.copy()]
        current_pos = start_pos.copy()
        dt_plan = 2
        max_steps = 200

        for step in range(max_steps):
            # 计算势场力
            force = self.compute_potential_field(current_pos, goal_pos)

            # 限制最大步长
            force_norm = np.linalg.norm(force)
            if force_norm > 5.0:
                force = force / force_norm * 5.0

            # 更新位置
            current_pos = current_pos + force * dt_plan
            path.append(current_pos.copy())

            # 检查是否到达目标
            if np.linalg.norm(current_pos - goal_pos) < 1.0:
                print(f"路径规划完成，步数: {step}")
                break

        return np.array(path)


# 全局路径规划器实例
path_planner = CBFPathPlanner()


class CBFController:
    def __init__(self):
        self.v_safe_margin = 0.3
        self.obstacles = [
            {'position': obs1_start_position, 'radius': obs_radius, 'end_pos': obs1_end_position},
            {'position': obs2_start_position, 'radius': obs_radius, 'end_pos': obs2_end_position}
        ]
        self.safe_distance = 3
        self.gamma = 1.8

    def cbf_constraints(self, current_pos, current_vel, control_input, current_time):
        """生成CBF约束 - 用于实时避障"""
        constraints = []
        current_pos = current_pos.flatten()

        # 根据当前时间更新动态障碍物位置
        progress = current_time / T
        for obs in self.obstacles:
            obs_pos = obs['position'] + progress * (obs['end_pos'] - obs['position'])
            to_obs = current_pos - obs_pos
            distance = np.linalg.norm(to_obs)

            influence_distance = self.safe_distance * obs_radius

            if distance < influence_distance:
                hx = distance - (self.safe_distance * obs_radius)
                h_grad = to_obs / distance
                cbf_constraint = h_grad @ control_input >= -self.gamma * hx
                constraints.append(cbf_constraint)

        return constraints

    def optimize_control(self, current_pos, current_vel, nominal_control, current_time):
        """CBF优化求解安全控制量"""
        u_opt = cp.Variable(2)
        cost = cp.sum_squares(u_opt - nominal_control.flatten())

        # 获取CBF约束
        cbf_constraints = self.cbf_constraints(current_pos, current_vel, u_opt, current_time)

        # 控制输入约束
        control_constraints = [
            cp.norm(u_opt, 2) <= 300,
            u_opt[0] >= -200, u_opt[1] >= -200,
            u_opt[0] <= 200, u_opt[1] <= 200
        ]

        all_constraints = cbf_constraints + control_constraints

        prob = cp.Problem(cp.Minimize(cost), all_constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status == cp.OPTIMAL:
                safe_control = u_opt.value.reshape(2, 1)
                print(f"实时避障安全速度: [{safe_control[0, 0]:.3f}, {safe_control[1, 0]:.3f}]")
                return safe_control
            else:
                print(f"CBF优化失败，使用名义控制")
                return nominal_control
        except Exception as e:
            print(f"CBF求解异常: {e}，使用名义控制")
            return nominal_control




cbf_controller = CBFController()
magnetic_model = model.magnetic_motion_model(dt=dt, position=[[8], [0]], vel=[[0.0001],[0.0001]], theta = np.pi/2, current_limit = 10)


def generate_follow_path():
    global magnetic_model

    # 设置起点和终点
    start_pos = magnetic_model.return_position().reshape(-1)  # 机器人起始位置
    goal_pos = goals_pos  # 目标位置

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


# def generate_follow_path():
#     global magnetic_model
#
#
#
#     # 路近点跟踪
#     global r
#     magnetic_model.ref_time = np.arange(t_start, t_end + dt, dt)
#
#     # 组合成路径数组 (N×2的矩阵)
#     N = len(magnetic_model.ref_time)  # 总点数
#
#     # 归一化时间比例
#     tn = magnetic_model.ref_time / T
#     magnetic_model.obs1_x = obs1_start_position[0] + (obs1_end_position[0] - obs1_start_position[0]) * tn
#     magnetic_model.obs1_y = obs1_start_position[1] + (obs1_end_position[1] - obs1_start_position[1]) * tn
#
#     magnetic_model.obs2_x = obs2_start_position[0] + (obs2_end_position[0] - obs2_start_position[0]) * tn
#     magnetic_model.obs2_y = obs2_start_position[1] + (obs2_end_position[1] - obs2_start_position[1]) * tn
#
#     n_quarter = N // 4  # 每段 1/4 时间的点数
#
#
#     magnetic_model.ref_x_1 = r * np.cos(np.pi/2)
#     magnetic_model.ref_x_2 = r * np.cos(np.pi)
#     magnetic_model.ref_x_3 = r * np.cos(3 * np.pi / 2)
#     magnetic_model.ref_x_4 = r * np.cos(2 * np.pi)
#     magnetic_model.ref_y_1 = r * np.sin(np.pi/2)
#     magnetic_model.ref_y_2 = r * np.sin(np.pi)
#     magnetic_model.ref_y_3 = r * np.sin(3 * np.pi / 2)
#     magnetic_model.ref_y_4 = r * np.sin(2 * np.pi)
#
#     magnetic_model.ref_vx = np.zeros(N)
#     magnetic_model.ref_vy = np.zeros(N)
#     magnetic_model.ref_ax = np.zeros(N)
#     magnetic_model.ref_ay = np.zeros(N)
#
#     # 构造 ref_x
#     magnetic_model.ref_x = np.zeros(N)
#     magnetic_model.ref_x[:n_quarter] = magnetic_model.ref_x_1  # 前 1/4 时间
#     magnetic_model.ref_x[n_quarter:2 * n_quarter] = magnetic_model.ref_x_2  # 接下来的 1/4
#     magnetic_model.ref_x[2 * n_quarter:3 * n_quarter] = magnetic_model.ref_x_3  # 再接下来的 1/4
#     magnetic_model.ref_x[3 * n_quarter:] = magnetic_model.ref_x_4  # 最后 1/4
#     magnetic_model.ref_y = np.zeros(N)
#     magnetic_model.ref_y[:n_quarter] = magnetic_model.ref_y_1
#     magnetic_model.ref_y[n_quarter:2 * n_quarter] = magnetic_model.ref_y_2
#     magnetic_model.ref_y[2 * n_quarter:3 * n_quarter] = magnetic_model.ref_y_3
#     magnetic_model.ref_y[3 * n_quarter:] = magnetic_model.ref_y_4
#
#     magnetic_model.ref_path = np.column_stack((magnetic_model.ref_x, magnetic_model.ref_y))
#     magnetic_model.ref_velpath = np.column_stack((magnetic_model.ref_vx, magnetic_model.ref_vy))
#     magnetic_model.ref_accelpath = np.column_stack((magnetic_model.ref_ax, magnetic_model.ref_ay))
#
#     print("magnetic_model.ref_path:",magnetic_model.ref_path)


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
    magnetic_model.theta = calculate_angle(magnetic_model.return_position().reshape(-1), magnetic_model.ref_path[0])- 1 / 2 * np.pi
    # magnetic_model.vel = np.array([[0.001*np.cos(magnetic_model.theta)], [0.001*np.sin(magnetic_model.theta)]])
    u_x, u_y, u_z = np.sin(alpha), -np.cos(alpha), 0  # 假设 x 方向单位向量
    v_x, v_y, v_z = np.cos(alpha) * np.cos(magnetic_model.theta), np.sin(alpha) * np.cos(magnetic_model.theta), -np.sin(magnetic_model.theta)  # 转向
    # cHx = amplitude * (u_x * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_x * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    # cHz = amplitude * (u_y * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_y * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    # cHy = amplitude * (u_z * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_z * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHx = amplitude * (u_x * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_x * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHz = amplitude * (u_y * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_y * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    cHy = amplitude * (u_z * np.cos(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]) - v_z * np.sin(2 * np.pi * 0.2 * magnetic_model.ref_time[cnt]))
    # 生成对应的电流
    IHx = cHx / ((4 / 5) ** (3 / 2) * miu_0 * 297 / 0.236)
    IHz = cHz / ((4 / 5) ** (3 / 2) * miu_0 * 202 / 0.162)
    IHy = cHy / ((4 / 5) ** (3 / 2) * miu_0 * 129 / 0.1)

    # cnt += 1
    return IHx, IHy, IHz

part2 = 0
def motion_control(i_motion, dt):
    global magnetic_model,part2,beita,cost_state,L_state,exceed_flag_state,Ks,last_current_error_robust

    if i_motion < len(magnetic_model.ref_time):
        time, xd, yd, vxd, vyd, axd, ayd = magnetic_model.ref_time[i_motion], magnetic_model.ref_x[i_motion], magnetic_model.ref_y[i_motion], magnetic_model.ref_vx[i_motion], magnetic_model.ref_vy[i_motion], magnetic_model.ref_ax[i_motion], magnetic_model.ref_ay[i_motion]
        beita = np.zeros((4, 1))
        if time < 0.07:
            last_current_error_robust = np.zeros((4, 1))


        magnetic_model.path_log.append(magnetic_model.position)
        if i_motion == 0:
            kf.statePost = np.array([[magnetic_model.position[0,0]], [magnetic_model.position[1,0]], [magnetic_model.vel[0,0]], [magnetic_model.vel[1,0]]], dtype=np.float32)
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
        Kd = 0.5 # 速度增益

        desired_vel_raw = Kp * error + Kd * dot_error

        desired_vel = np.linalg.norm(desired_vel_raw)
        desired_omega=desired_vel/2
        I = 1e-4  # 转动惯量 [kg·mm²]
        current_omega = np.linalg.norm(magnetic_model.vel) / 2
        # print("现在的omega：°",current_omega*57.3)

        #CBF控制
        u_safe=cbf_controller.optimize_control(np.array(magnetic_model.return_position()),
                                        np.array([[vxd], [vyd]]), desired_vel_raw,time)
        print("名义控制：",desired_vel_raw)
        print("CBF计算后的速度",u_safe)

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
        T_inv21 = 0.02/0.5
        T_inv22=0.5

        #1.22e-9是溶液阻力系数
        if i_motion > 0:
            alpha_omega = (desired_omega - current_omega) / dt #角加速度

            error_robust=current_omega-theta_control
            theta_big_change=-dao*sigmoid(current_omega)*(error_robust*P*(1/1e-4)-miu_v*sigmoid(current_omega)*theta_big_pre*(1/1e-4)*P*(-0.1)*(1/1e-4))
            theta_big=theta_big_pre+theta_big_change*dt
            uad=theta_big*sigmoid(current_omega)
            #print("计算出的uad",uad)
            force1=alpha_omega*2*48
            desired_torque = I * alpha_omega+1.22e-9*desired_omega
            #print("计算出的处理前扭矩", desired_torque)
            actual_torque =T_inv21*force1- T_inv22 * desired_torque
            print("估计后的扭矩：",actual_torque-uad)
            finally_torque=actual_torque*1.3-uad
        else:
            actual_torque = np.array([[0.0], [0.0]])
            finally_torque=0

        desired_field_strength = np.linalg.norm(finally_torque) / (magnetic_model.Magnetic* np.sin(theta_control))
        #print("计算出的处理前扭矩", desired_torque)
        print("计算出的actual_torque",actual_torque)
        print("计算出的总的磁场幅值",desired_field_strength)
        #desired_field_strength = np.clip(desired_field_strength, 0.0011, 0.005)
        desired_field_strength = np.clip(desired_field_strength, 0, 0.0085)
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
        magnetic_model.input_smith_log.append(input)    #   记录每一时刻的加上smith补偿的理论输入

        # 这边是动力学部分，暂时不考虑更改

        omega_vel = np.linalg.norm((magnetic_model.vel + magnetic_model.accel * dt))
        # omega应该是指磁体要滚得多块，这边对应磁体边长上一点的线速度·
        # (np.linalg.norm)用来求取二范数，这边由于vel里头是velx和vely，所以取二范数
        omega_vel = omega_vel / (2 * np.pi * 2)
        # 将线速度转化为角速度，除以2pi*r，这边的r应该是2mm的意思，在更换磁体的时候需要修改
        # print("orignal omgea: ", omega_vel)
        omega_vel = np.clip(omega_vel, 0, 0.4)


        u_x, u_y, u_z = np.sin(alpha), -np.cos(alpha), 0  # 假设 x 方向单位向量
        v_x, v_y, v_z = np.cos(alpha) * np.cos(theta_applied), np.sin(alpha) * np.cos(theta_applied), -np.sin(theta_applied)  # 转向
        print("theta情况",theta_applied)

        cHx = desired_field_strength * (u_x * np.cos(theta_control * magnetic_model.ref_time[i_motion]) - v_x * np.sin(theta_control * magnetic_model.ref_time[i_motion]))
        cHz = desired_field_strength * (u_y * np.cos(theta_control * magnetic_model.ref_time[i_motion]) - v_y * np.sin(theta_control * magnetic_model.ref_time[i_motion]))
        cHy = desired_field_strength * (u_z * np.cos(theta_control * magnetic_model.ref_time[i_motion]) - v_z * np.sin(theta_control * magnetic_model.ref_time[i_motion]))

        # 生成对应的电流
        IHx = cHx / ((4 / 5) ** (3 / 2) * miu_0 * 297 / 0.236)
        IHz = -cHz / ((4 / 5) ** (3 / 2) * miu_0 * 202 / 0.162)
        IHy = -cHy / ((4 / 5) ** (3 / 2) * miu_0 * 129 / 0.1)

        magnetic_model.current_log.append([IHx, IHy, IHz])

        theta_pre = theta_applied

        # # 进行限幅
        # IH = jkm
        # current_input = [max(min(i, magnetic_model.current_limit), -magnetic_model.current_limit) for i in IH]

        # magnetic_model.current_log.append(current_input)
        # magnetic_model.error_log.append(error)
        # magnetic_model.path_log.append(magnetic_model.return_position())
        # magnetic_model.vel_log.append(magnetic_model.return_vel())

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

    # 创建一个新的Excel工作簿
    # workbook = openpyxl.Workbook()
    # # 选择要操作的工作表
    # sheet = workbook.active
    # draw.draw_log(magnetic_model.ref_time, magnetic_model.ref_x, magnetic_model.ref_y, magnetic_model.ref_v, magnetic_model.ref_yaw, magnetic_model, magnetic_ppc)