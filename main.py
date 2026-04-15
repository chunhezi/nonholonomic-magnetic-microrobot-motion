import sys
import time
from PyQt6 import QtWidgets
from PyQt6.QtGui import *
from Widgets_window import Ui_Form
import cv2
import numpy as np
import UartSerial
import PowerSupply
import Drive
import motion
import math
# 在主文件开头导入
# from motion import CBFController
import threading
from collections import deque
import pandas as pd


'''
在 motion_control 函数末尾添加一个专门的记录函数：我要记录机器人的速度，角速度，位置，
朝向角theta，速度、角速度和位置的误差，
XYZ三轴电流，扭矩，自适应参数theta_big，CBF求解出来的速度，旋转磁场角度与磁矩的角度差
tensorboard --logdir=G:/postgraduate/anquan/CBF_robu_lujin_plus_kf
tensorboard --logdir=G:/postgraduate/anquan/video/test/13
G:\postgraduate\huiyi\video\1
tensorboard --logdir=G:\postgraduate\huiyi\video\3
'''

# 全局变量
cbf_controller = motion.CBFController()

port_state = False
motion_state = False
camera_state = False
video_state = True
i_motion = 0
pts = deque(maxlen=1240000)
square_state = False
square_position = (0, 0)
square_box = None
min_area = 1150
max_area = 1500
# 添加正方形检测的面积范围
square_min_area = 100000  # 根据实际正方形大小调整
square_max_area = 900000  # 根据实际正方形大小调整
Camera_index = 0

# 障碍物检测状态
obstacle_detected = False  # 新增：障碍物检测状态标志
fixed_obstacles = []  # 新增：固定的障碍物信息

# 这边定义了串口通讯的东西，应该是初始化的部分
uart = UartSerial.UartSerial()
power = PowerSupply.PowerSupply()
magnetic_drive = Drive.Drive(uart, power)

T = 2 * np.pi / 0.2  # 运动周期 (s)

# =====================
# 生成轨迹数据
# =====================
t_start = 0  # 起始时间 (s)
t_end = T  # 结束时间 (s) - 完整周期
dt = 0.035  # 时间步长 (s)
dt_ms = 35
ref_time = np.arange(t_start, t_end + dt, dt)


class WinForm(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.comboBox_baud.addItem("921600", "921600")
        self.comboBox_baud.addItem("460800", "460800")
        self.comboBox_baud.addItem("230400", "230400")
        self.comboBox_baud.addItem("1000000", "1000000")
        self.comboBox_baud.addItem("38400", "38400")
        self.comboBox_baud.addItem("19200", "19200")
        self.comboBox_baud.addItem("9600", "9600")
        self.comboBox_baud.addItem("4800", "4800")
        self.comboBox_baud.setCurrentIndex(3)
        # 初始化槽函数
        self.slot_init()
        # 扫描串口
        self.refresh_serial_ports()
        # 初始化摄像头线程
        self.thread_camera = threading.Thread(target=self.detect)

    def slot_init(self):
        pass
        # 串口按钮
        self.button_open_serial.clicked.connect(self.serial_open_off)
        # 关闭串口按钮
        self.button_refresh_serial.clicked.connect(self.refresh_serial_ports)
        # 退出
        self.button_exit.clicked.connect(app.quit)

        # 摄像头函数
        self.button_camera.clicked.connect(self.camera_event)
        # path motion
        self.button_motion.clicked.connect(self.motion_event)

        # up
        self.button_up.clicked.connect(magnetic_drive.motion_up)
        # down
        self.button_down.clicked.connect(magnetic_drive.motion_down)
        # left
        self.button_left.clicked.connect(magnetic_drive.motion_left)
        # right
        self.button_right.clicked.connect(magnetic_drive.motion_right)
        # stop
        self.button_stop.clicked.connect(magnetic_drive.motion_stop)

    # 串口检测
    def refresh_serial_ports(self):
        _ports = uart.get_all_port()
        # print(_ports)
        self.comboBox_port.clear()
        if len(_ports) == 0:
            self.comboBox_port.addItem('')
        else:
            for item in _ports:
                self.comboBox_port.addItem(item)

    def serial_open_off(self):
        global port_state
        str = self.button_open_serial.text()
        #   这个命令存储了选定的是哪个串口以及是哪个波特率
        port_name = self.comboBox_port.currentText()
        baud_rate = int(self.comboBox_baud.currentText())

        # 这边要改一下，if的判断要改成判断是否使能成功的
        if str == '关闭串口':
            # 关闭电源
            ret = magnetic_drive.uninit_power()
            # port_close也是0f写的，具体的点进去看就行
            if ret:
                self.button_open_serial.setText('打开串口')
                port_state = False
            else:
                print("Close Uart Fail!!")

        # 这边也要改一下，if的判断要改成判断是否使能成功的
        if str == '打开串口':
            if uart.is_port_open():  # port_name, baud_rate):
                # 这边决定打开的是哪个串口和波特率，也就是在这一步把port_name和baud_rate传递给uarserial中的port的
                ret = uart.try_port_open(port_name, baud_rate)
                if ret:
                    self.button_open_serial.setText('关闭串口')
                    # 初始化电流
                    magnetic_drive.init_power()
                    port_state = True
                else:
                    print("Client Fail!!")
            else:
                print("Client success!!")

    def camera_event(self):
        global camera_state, obstacle_detected, fixed_obstacles
        str = self.button_camera.text()

        if str == '打开相机':
            # start the thread
            if self.thread_camera.is_alive():
                camera_state = True
            else:
                camera_state = True
                self.thread_camera.start()
            self.button_camera.setText('关闭相机')
        elif str == '关闭相机':
            camera_state = False
            self.button_camera.setText('打开相机')
            # 重置障碍物检测状态（可选）
            # obstacle_detected = False
            # fixed_obstacles = []

    def motion_event(self):
        global i_motion, motion_state
        if not motion_state:
            print("start motion")
            motion_state = True
            i_motion = 0
            self.button_motion.setText('stop motion')
        else:
            print("stop motion")
            motion_state = False
            self.button_motion.setText('start motion')
            motion.motion_log()

    def detect_square(self, contours):
        """检测符合条件的正方形轮廓"""
        square_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积过滤 - 只检测中等大小的轮廓
            if area < square_min_area or area > square_max_area:
                continue

            # 多边形逼近检测四边形
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 检查是否是四边形
            if len(approx) == 4:
                # 检查是否是凸边形
                if cv2.isContourConvex(approx):
                    # 获取最小外接矩形
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]

                    # 计算宽高比（接近1表示是正方形）
                    aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0

                    # 如果是比较接近正方形的形状
                    if aspect_ratio > 0.8:
                        square_candidates.append((contour, rect, area))

        # 返回面积最大的正方形候选
        if square_candidates:
            return max(square_candidates, key=lambda x: x[2])
        return None, None, 0

    def detect_obstacles_in_square(self, cv_img, square_box):
        """在正方形区域内检测暗色障碍物，保存边界点"""
        if square_box is None:
            return []

        # 创建正方形区域的掩码
        mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [square_box], 255)

        # 提取正方形区域
        square_region = cv2.bitwise_and(cv_img, cv_img, mask=mask)

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(square_region, cv2.COLOR_BGR2HSV)

        obstacles = []

        # 针对暗色障碍物的HSV范围
        dark_lower = np.array([0, 50, 0])
        dark_upper = np.array([180, 255, 80])

        # 创建暗色区域的掩码
        mask_dark = cv2.inRange(hsv, dark_lower, dark_upper)

        # 形态学操作去除噪声
        kernel = np.ones((7, 7), np.uint8)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel)

        # 查找暗色障碍物轮廓
        dark_contours, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in dark_contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 200000:
                # 获取障碍物边界框
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                # 计算轮廓特征
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # 获取边界点（轮廓点）
                boundary_points = [tuple(point[0]) for point in contour]  # 提取轮廓点 (x, y)

                obstacles.append({
                    'center': center,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'circularity': circularity,
                    'contour': contour,
                    'boundary_points': boundary_points,  # 只保存边界点
                    'point_count': len(boundary_points)
                })

        # 保存边界点到Excel
        if obstacles:
            try:
                # 准备边界点数据
                boundary_data = []
                for i, obstacle in enumerate(obstacles):
                    # 为每个障碍物的每个边界点创建一行数据
                    for j, point in enumerate(obstacle['boundary_points']):
                        # 转换为相对坐标
                        rel_x = (point[0] - square_position[0]) / 10.0
                        rel_y = (point[1] - square_position[1]) / 10.0

                        boundary_data.append({
                            'obs_id': i + 1,
                            'point_id': j + 1,
                            'boundary_X': point[0],  # 原始边界点X
                            'boundary_Y': point[1],  # 原始边界点Y
                            'rel_X': round(rel_x, 2),  # 相对X坐标
                            'rel_Y': round(rel_y, 2),  # 相对Y坐标
                            'center_X': obstacle['center'][0],
                            'center_Y': obstacle['center'][1],
                            '障碍物面积': obstacle['area'],
                            '障碍物圆形度': round(obstacle['circularity'], 3)
                        })

                # 创建DataFrame并保存
                df = pd.DataFrame(boundary_data)
                filename = "boundary_points.xlsx"
                df.to_excel(filename, index=False)

                print(f"边界点数据已保存到: {filename}")
                print(f"共检测到 {len(obstacles)} 个障碍物")
                print(f"总边界点数: {sum(obs['point_count'] for obs in obstacles)}")

            except Exception as e:
                print(f"保存Excel失败: {e}")

        return obstacles

    def draw_fixed_obstacles(self, cv_img):
        """绘制固定的障碍物信息"""
        global fixed_obstacles

        for obstacle in fixed_obstacles:
            center = obstacle['center']
            bbox = obstacle['bbox']
            area = obstacle['area']

            # 绘制轮廓 - 使用固定的颜色
            cv2.drawContours(cv_img, [obstacle['contour']], -1, (18, 153, 255), 2)

            # 显示障碍物信息
            info_text = f"Fixed: {area:.0f}px"
            cv2.putText(cv_img, info_text,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (18, 153, 255), 2)

    def detect(self):
        global square_position, square_state, camera_state, i_motion
        global transform_state, transform_mile, square_state, square_position, square_box
        global obstacle_detected, fixed_obstacles  # 新增全局变量


        cap = cv2.VideoCapture(Camera_index)

        # 设置摄像头的帧率为60fps 设置摄像头的分辨率为1080p（1920x1080）
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        print('HEIGHT:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('WIDTH:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('FPS:', round(cap.get(cv2.CAP_PROP_FPS)))

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_output = cv2.VideoWriter('output.avi', fourcc, 30.0, (1920, 1080))
        if not video_output.isOpened():
            print("错误：VideoWriter 未打开！请检查编码器或路径。")
            exit()

        # 计算帧率，终点
        time_mark = time.perf_counter()
        motion.generate_follow_path(np.array([2, -20]))
        while True:
            if not camera_state:
                break

            # 计算帧率
            deltaT = (time.perf_counter() - time_mark) * 1000
            print('time:%.2f ms' % deltaT)
            time_mark = time.perf_counter()

            # 读取图像
            ret, cv_img = cap.read()
            cv_img = cv2.flip(cv_img, 1)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)  # 二值化
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 卷积核
            eroded = cv2.erode(binary, kernel, iterations=2)  # 腐蚀
            contours, hierarchy = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 边缘检测

            if contours:
                # 正方形检测替代圆形检测
                if not square_state:
                    # 使用改进的正方形检测方法
                    square_contour, square_rect, square_area = self.detect_square(contours)

                    if square_contour is not None:
                        # 获取正方形边界点
                        box_points = cv2.boxPoints(square_rect)
                        box_points = np.int32(box_points)

                        # 计算中心点
                        square_center = np.mean(box_points, axis=0).astype(int)
                        square_position = [square_center[0], square_center[1]]

                        # 计算边长
                        width, height = square_rect[1]
                        square_side = int((width + height) / 2)

                        print("Square center: ", square_position)
                        print("Square side length: ", square_side)
                        print("Square area: ", square_area)  # 打印面积用于调试
                        square_state = True
                        square_box = box_points
                else:
                    pass

                filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
                for c in filtered_contours:
                    cv2.drawContours(cv_img, [c], -1, (255, 0, 0), 2)  # 蓝色表示符合条件的轮廓
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    robot_center = (int(x), int(y), deltaT / 1000)
                    # 使用正方形中心作为参考坐标系
                    motion.update_status(
                        np.array([[(x - square_position[0]) / 10], [(y - square_position[1]) / 10]]))
                    print("robot_center: ", robot_center)
                    error = motion.magnetic_model.error
                    print("error: ", error)
                    print("detect position: ", motion.return_current_position().reshape(-1))
                    if motion_state:
                        pts.append(robot_center)

                if square_state:
                    # 绘制正方形边界
                    cv2.drawContours(cv_img, [square_box], -1, (0, 255, 0), thickness=2)
                    if not obstacle_detected:
                        # 在正方形内检测暗色障碍物
                        obstacles = self.detect_obstacles_in_square(cv_img, square_box)

                        if obstacles:  # 如果检测到障碍物
                            fixed_obstacles = obstacles  # 固定障碍物信息
                            obstacle_detected = True  # 设置标志位
                            print("=== 障碍物检测完成，信息已固定 ===")
                            for obstacle in fixed_obstacles:
                                center = obstacle['center']
                                area = obstacle['area']
                                print(f"固定障碍物: 中心{center}, 面积{area:.0f}")

                    # 绘制固定的障碍物
                    self.draw_fixed_obstacles(cv_img)
            #画终点


                #motion.generate_follow_path(np.array([-12, -33]))
                cv2.circle(cv_img, np.array([-10, -30]), 10, (96,48,176), -1)
                if motion_state:
                    ################画障碍物##################
                    for i, obstacle in enumerate(cbf_controller.obstacles):
                        radius_obs = int(10 * obstacle['radius'])
                        radius_safe = int(obstacle['radius'] * cbf_controller.safe_distance * 10)

                        if i == 0:  # 第一个障碍物
                            img_x = int(square_position[0] + 10 * motion.magnetic_model.obs1_x[i_motion])
                            img_y = int(square_position[1] + 10 * motion.magnetic_model.obs1_y[i_motion])

                        elif i == 1:  # 第二个障碍物
                            img_x = int(square_position[0] + 10 * motion.magnetic_model.obs2_x[i_motion])
                            img_y = int(square_position[1] + 10 * motion.magnetic_model.obs2_y[i_motion])

                        # 障碍物。颜色color数值是BGR，但是在搜索出来的确实RGB
                        cv2.circle(cv_img, (img_x, img_y), radius_obs, (128, 128, 240), -1)

                        # 画描边. 每15度画一段10度的虚线
                        for angle_deg in range(0, 360, 20):  # 每15度一个循环
                            # 画10度的线段
                            start_angle = math.radians(angle_deg)
                            end_angle = math.radians(angle_deg + 10)

                            start_x = int(img_x + radius_obs * math.cos(start_angle))
                            start_y = int(img_y + radius_obs * math.sin(start_angle))
                            end_x = int(img_x + radius_obs * math.cos(end_angle))
                            end_y = int(img_y + radius_obs * math.sin(end_angle))

                            start_xsafe = int(img_x + radius_safe * math.cos(start_angle))
                            start_ysafe = int(img_y + radius_safe * math.sin(start_angle))
                            end_xsafe = int(img_x + radius_safe * math.cos(end_angle))
                            end_ysafe = int(img_y + radius_safe * math.sin(end_angle))

                            cv2.line(cv_img, (start_x, start_y), (end_x, end_y), (0, 0, 0), 2)
                            cv2.line(cv_img, (start_xsafe, start_ysafe), (end_xsafe, end_ysafe), (120, 128, 250), 2)

                    # # 画跟踪点
                    # cv2.circle(cv_img, np.array([square_position[0] + 10 * int(motion.magnetic_model.ref_x[i_motion]),
                    #                              square_position[1] + 10 * int(motion.magnetic_model.ref_y[i_motion])]),
                    #            4, (0, 0, 255), -1)

                    # 画轨迹点
                    for i in range(1, len(pts)):
                        if pts[i - 1][:2] is None or pts[i][:2] is None:
                            continue
                        cv2.line(cv_img, pts[i - 1][:2], pts[i][:2], (0, 0, 255), thickness=1)

            # 录像
            if video_state:
                video_output.write(cv_img)  # 保存视频

            # 裁剪
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(cv_img, (640, 360))
            frame = QImage(resized_img, 640, 360, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.label_show_camera.setPixmap(pix)

            if motion_state:
                #   这边是要改的，要同时控制六个线圈，得加上IHx1, IHy1, IHz1，IHx2, IHy2, IHz2
                IHx, IHy, IHz = motion.motion_control(i_motion, deltaT / 1000)
                print("Current: ", IHx, IHy, IHz)
                i_motion = i_motion + 1
            else:
                # IHx, IHy, IHz = motion.adjust_angle()  # 这边是那个小作弊的方式
                IHx, IHy, IHz = 0, 0, 0

            # 驱动电流
            # 驱动部分的修改主要集中在这部分和Drive.py部分
            if port_state:  # and motion_state:
                # 这边也是要改的，VHx决定了电压的方向，从而产生正负的电流，设置的时候应该设置的都是绝对值，
                # 但对于有驱动的情况，无需这样设置，直接用占空比产生负电
                if IHx > 10:
                    IHx = 10
                elif IHx < -10:
                    IHx = -10
                if IHy > 10:
                    IHy = 10
                elif IHy < -10:
                    IHy = -10
                if IHz > 10:
                    IHz = 10
                elif IHz < -10:
                    IHz = -10
                magnetic_drive.set_currents(IHx, IHx, IHy, IHy, IHz, IHz)

        cap.release()
        # 录像
        if video_state:
            video_output.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = WinForm()
    form.show()
    sys.exit(app.exec())