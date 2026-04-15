import sys
import time
from PyQt6 import QtCore, QtGui, QtWidgets
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
from motion import CBFController
import threading
from collections import deque

# 全局变量
cbf_controller = motion.CBFController()

port_state = False
motion_state = False
camera_state = False
video_state = True
i_motion = 0
pts = deque(maxlen=1240000)
circle_state = False
circle_position = (0, 0)
transform_state = False
transform_mile = 10
min_area = 850
max_area = 1550
Camera_index = 0

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
# ref_x = 10 * np.cos(0.2 * ref_time).astype(int) + circle_position
# ref_y = 10 * np.sin(0.2 * ref_time).astype(int)
#
# # 组合成路径数组 (N×2的矩阵)
# ref_path = np.column_stack((ref_x, ref_y)).tolist()

# 录像
# if video_state:
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# video_output = cv2.VideoWriter('output.avi', fourcc, 30.0, (1920, 1080))
# if not video_output.isOpened():
#     print("错误：VideoWriter 未打开！请检查编码器或路径。")
#     exit()

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
        global camera_state
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

    def detect(self):
        global circle_position, circle_state, camera_state, i_motion
        global transform_state, transform_mile, circle_state, circle_position
        motion.generate_follow_path()
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

        # 计算帧率
        time_mark = time.perf_counter()

        while True:
            if not camera_state:
                break

            # 计算帧率
            #   这一块说实话没能看懂为什么要出deltaT
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
                # 找到面积最大的轮廓
                if not circle_state:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # 画出该轮廓
                    # cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    circle_position = [int(x), int(y)]
                    circle_radius = int(radius)
                    # cv2.circle(cv_img, circle_position, circle_radius, (0, 0, 255))
                    print("img center: ", circle_position)
                    circle_state = True
                else:
                    cv2.circle(cv_img, circle_position, 3, (0, 0, 255), -1)

                filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
                for c in filtered_contours:
                    #     # 遍历所有轮廓，计算并输出面积
                    #     for i, c in enumerate(contours):
                    #         area = cv2.contourArea(c)
                    #         print(f"轮廓 {i} 的面积: {area}")
                    cv2.drawContours(cv_img, [c], -1, (255, 0, 0), 2)  # 蓝色表示符合条件的轮廓
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    robot_center = (int(x), int(y), deltaT / 1000)
                    # motion.update_status(np.array([[x / 10], [y / 10]]))
                    motion.update_status(
                        np.array([[(x - circle_position[0]) / 10], [(y - circle_position[1]) / 10]]))
                    print("robot_center: ", robot_center)
                    error = motion.magnetic_model.error
                    print("error: ", error)
                    print("detect position: ", motion.return_current_position().reshape(-1))
                    if motion_state:
                        pts.append(robot_center)

                # draw path
                if circle_state:
                    # 这个是一开始画的绿色的圆
                    # cv2.circle(cv_img, circle_position, 4, (0, 0, 255), -1)
                    cv2.circle(cv_img, circle_position, circle_radius, (0, 255, 0), thickness=2)

                if motion_state:
                    ################画障碍物##################
                    for i, obstacle in enumerate(cbf_controller.obstacles):
                        # img_x = int(circle_position[0] + 10 * obstacle['position'][0])
                        # img_y = int(circle_position[1] + 10 * obstacle['position'][1])
                        radius_obs = int(10 * obstacle['radius'])
                        radius_safe=int(obstacle['radius'] * cbf_controller.safe_distance * 10)

                        if i == 0:  # 第一个障碍物
                            img_x = int(circle_position[0] + 10 * motion.magnetic_model.obs1_x[i_motion])
                            img_y = int(circle_position[1] + 10 * motion.magnetic_model.obs1_y[
                                i_motion])  # 修正：circle_position[1]

                        elif i == 1:  # 第二个障碍物
                            img_x = int(circle_position[0] + 10 * motion.magnetic_model.obs2_x[i_motion])
                            img_y = int(circle_position[1] + 10 * motion.magnetic_model.obs2_y[
                                i_motion])  # 修正：circle_position[1]

                        #障碍物。颜色color数值是BGR，但是在搜索出来的确实RGB
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

                            start_xsafe=int(img_x + radius_safe * math.cos(start_angle))
                            start_ysafe=int(img_y + radius_safe * math.sin(start_angle))
                            end_xsafe = int(img_x + radius_safe * math.cos(end_angle))
                            end_ysafe = int(img_y + radius_safe * math.sin(end_angle))

                            cv2.line(cv_img, (start_x, start_y), (end_x, end_y), (0, 0, 0), 2)
                            cv2.line(cv_img, (start_xsafe, start_ysafe), (end_xsafe, end_ysafe), (120, 128, 250), 2)


                    # cv2.circle(cv_img, np.array([circle_position[0] + 10 * 8,
                    #                              circle_position[1] + 10 * 2]), 30, (0, 191, 225), -1)


                    #画跟踪点
                    cv2.circle(cv_img, np.array([circle_position[0] + 10 * int(motion.magnetic_model.ref_x[i_motion]),
                                                 circle_position[1] + 10 * int(motion.magnetic_model.ref_y[i_motion])]),
                               4, (0, 0, 255), -1)

                    #画轨迹点
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
