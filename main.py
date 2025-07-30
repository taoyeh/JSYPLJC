import atexit
import sys
import time

import cv2
import numpy as np
from Window import Ui_Form
from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from config import config
import myfarame
from playsound import playsound
from threading import Thread

# 自适应阈值计算器
class AdaptiveThresholdCalculator:
    def __init__(self, init_eye_thresh=0.26):
        self.eye_thresh = init_eye_thresh
        self.eye_buffer = []
        self.buff_size = 30

        self.eye_adjust_factor = 0.05
        self.min_eye_thresh = 0.15
        self.max_eye_thresh = 0.35

    def update_thresholds(self, eye_ar):
        # 记录眼睛纵横比数据
        self.eye_buffer.append(eye_ar)
        # 维护缓冲区大小, 让其保持在 30 帧
        if len(self.eye_buffer) > self.buff_size:
            self.eye_buffer.pop(0)
        if len(self.eye_buffer) == self.buff_size:
            eye_mean = np.mean(self.eye_buffer)

            if eye_mean < self.eye_thresh - 0.1:
                self.eye_thresh = max(self.min_eye_thresh, self.eye_thresh-self.eye_adjust_factor)

            elif eye_mean > self.eye_thresh + 0.1:
                self.eye_thresh = min(self.max_eye_thresh, self.eye_thresh + self.eye_adjust_factor)


        return self.eye_thresh


# 疲劳检测类
class FatiguDetector:
    def __init__(self):
        self.thresholds_cal = AdaptiveThresholdCalculator()
        # 眼睛参数
        self.EYE_AR_FRAMES = 2

        # 嘴巴参数
        self.MAR_THRESH = 0.65
        self.MOUTH_AR_FRAME = 3

        # 疲劳参数
        self.FATIGUE_THRESHOLD = 0.30
        self.WARNING_THRESHOLD = 0.15

        # 上次评估时间
        self.last_check_time = time.time()
        # 上次的疲劳时间
        self.last_fatigue_time = 0
        # 上次的警告时间
        self.last_warning_time = 0

        # 初始化计数器
        self.reset_count()

        # 每隔2秒评估一次疲劳状态
        self.FATIGUE_CHECK_INTERVAL = 2.0
        # 状态维持 30 秒
        self.STATUS_MAINTAIN_DURATION = 30.0


    # 重置计数器
    def reset_count(self):
        self.eye_count = 0
        self.total_eye_closed = 0
        self.eye_cycle_count = 0
        self.thirty_sec_eye = 0  # 新增

        self.mouth_counter = 0
        self.total_mouth_open = 0
        self.mouth_cycle_count = 0
        self.thirty_sec_mouth = 0  # 新增


    # 检测疲劳状态, 'normal'、'warning'、“fatigue”
    def detec_fatigue(self, eyear, mouthar):
        # 获取自适应阈值
        self.EYE_AR_THRESH = self.thresholds_cal.update_thresholds(eyear)

        # print(f"EYE_AR{self.EYE_AR_THRESH}, eyear: {eyear}")

        if eyear < self.EYE_AR_THRESH:
            self.eye_count += 1
            self.eye_cycle_count +=1
        else:
            if self.eye_count>=self.EYE_AR_FRAMES:
                self.total_eye_closed += 1
                self.thirty_sec_eye += 1
            self.eye_count = 0

        if mouthar > self.MAR_THRESH:
            self.mouth_counter += 1
            self.mouth_cycle_count += 1
        else:
            if self.mouth_counter >= self.MOUTH_AR_FRAME:
                self.total_mouth_open += 1
                self.thirty_sec_mouth += 1
            self.mouth_counter = 0
        current_time = time.time()
        current_status = 'normal'

        if current_time - self.last_check_time >= self.FATIGUE_CHECK_INTERVAL:
            # 疲劳分数计算
            time_interval = max(0.1, current_time - self.last_check_time)
            fatigue_score = self.cal_fatigue_score(time_interval)
            print(f'疲劳分数: {fatigue_score}')

            # 复位
            self._reset_cycle_count()
            self.last_check_time = current_time

            # 根据疲劳状态，提醒
            if fatigue_score > self.FATIGUE_THRESHOLD:
                self.last_fatigue_time = current_time
                print("检测到疲劳状态")
            elif fatigue_score > self.WARNING_THRESHOLD:
                print("检测到警告状态")

        if current_time - self.last_fatigue_time < self.STATUS_MAINTAIN_DURATION:
            return 'fatigue'
        elif current_time - self.last_warning_time < self.STATUS_MAINTAIN_DURATION:
            return 'warning'
        else:
            return current_status


    # 计算疲劳分数
    def cal_fatigue_score(self, time_interval):
        est_frames = max(1, time_interval*30)
        eye_ratio = min(1, self.eye_cycle_count / est_frames)
        mouth_ratio = min(1, self.mouth_counter / est_frames)

        print(f'闭眼帧数{self.eye_cycle_count}, 张嘴帧数: {self.mouth_cycle_count}, 估计帧数: {est_frames}')
        print(f'闭眼比例: {eye_ratio}, 张嘴比例: {mouth_ratio}')

        return 0.8*eye_ratio + 0.2*mouth_ratio
    # 重置计数
    def _reset_cycle_count(self):
        self.eye_cycle_count = 0
        self.mouth_cycle_count = 0

# 摄像头控制器
class CameraController:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.timer = QTimer()
        self.cleaned = False

        # 记录疲劳的状态
        self.last_status = 'normal'
        self.fatigue_detector = FatiguDetector()

        # 声音警告
        self.warning_playing = False

        # 关闭的时候, 关闭摄像头资源
        self.parent_window.destroyed.connect(self.cleanup)
        atexit.register(self.cleanup)

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("摄像头未准备就绪")
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(20)
        except Exception as e:
            self.cleanup()
            raise

    def update_fatigue_status(self, status):
        status_text = {
            'normal': '清醒',
            'warning': '疲劳警告',
            'fatigue': '疲劳状态'
        }

        # print("=============" + status_text[status])

        # 设置文本 - 使用更简单的HTML结构
        self.parent_window.label_13.setText(
            f'<div style="text-align: center; font-size: 12pt; font-weight: bold;">'
            f'{status_text[status]}'
            f'</div>'
        )

        # 设置样式 - 统一使用setStyleSheet
        color_map = {
            'normal': 'rgb(85, 255, 127)',
            'warning': 'rgb(255, 170, 0)',
            'fatigue': 'rgb(255, 0, 0)'
        }

        self.parent_window.label_13.setStyleSheet(
            f"""
                QLabel {{
                    border: 3px solid rgb(66, 132, 198);
                    background-color: {color_map[status]};
                    color: black;  /* 确保文字颜色可见 */
                    padding: 5px;
                    min-width: 100px;  /* 确保最小宽度 */
                }}
                """
        )

        # 播放警告

    def _play_warning(self):
        if not self.warning_playing:
            self.warning_playing = True
            try:
                print("疲劳警告语音播报")
                playsound("voice.mp3")
            except Exception as e:
                print(e)
            finally:
                self.warning_playing = False

    def update_action_labels(self, action_label):
        """根据检测到的行为更新对应的UI标签"""
        # 行为检测计数器管理
        self.action_counter = getattr(self, 'action_counter', 0)

        # 重置计数器
        self.action_counter = max(0, self.action_counter - 1)

        # 更新检测到的行为标签
        if action_label == "phone":
            self.parent_window.label_16.setText(
                "<html><head/><body><p align='center'>"
                "<span style='font-size:10pt;font-weight:600;'>是</span></p ></body></html>")
            self.parent_window.label_16.setStyleSheet(
                "border:3px solid rgb(66, 132, 198);\n"
                "background-color:rgb(255, 0, 0)\n")
            self.action_counter = 5  # 重置计数器

        elif action_label == "drink":
            self.parent_window.label_17.setText(
                "<html><head/><body><p align='center'>"
                "<span style='font-size:10pt;font-weight:600;'>是</span></p ></body></html>")
            self.parent_window.label_17.setStyleSheet(
                "border:3px solid rgb(66, 132, 198);\n"
                "background-color:rgb(255, 0, 0)\n")
            self.action_counter = 5  # 重置计数器

        elif action_label == "smoke":
            self.parent_window.label_19.setText(
                "<html><head/><body><p align='center'>"
                "<span style='font-size:10pt;font-weight;600;'>是</span></p ></body></html>")
            self.parent_window.label_19.setStyleSheet(
                "border:3px solid rgb(66, 132, 198);\n"
                "background-color:rgb(255, 0, 0)\n")
            self.action_counter = 5  # 重置计数器

        # 当计数器达到0时重置所有标签
        if self.action_counter == 0:
            # 重置手机标签
            self.parent_window.label_16.setText(
                "<html><head/><body><p align='center'>"
                "<span style='font-size:10pt;font-weight:600;'>否</span></p ></body></html>")
            self.parent_window.label_16.setStyleSheet(
                "border:3px solid rgb(66, 132, 198);\n"
                "background-color:rgb(85, 255, 127)\n")

            # 重置喝水标签
            self.parent_window.label_17.setText(
                "<html><head/><body><p align='center'>"
                "<span style='font-size:10pt;font-weight:600;'>否</span></p ></body></html>")
            self.parent_window.label_17.setStyleSheet(
                "border:3px solid rgb(66, 132, 198);\n"
                "background-color:rgb(85, 255, 127)\n")

            # 重置吸烟标签
            self.parent_window.label_19.setText(
                "<html><head/><body><p align='center'>"
                "<span style='font-size:10pt;font-weight:600;'>否</span></p ></body></html>")
            self.parent_window.label_19.setStyleSheet(
                "border:3px solid rgb(66, 132, 198);\n"
                "background-color:rgb(85, 255, 127)\n")


    # 更新视频
    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)
            frame, ret = myfarame.frametest(frame)
            previous_status = self.last_status  # 使用上一次的状态作为默认值
            if ret:
                lab,eyear, mouthar = ret
                print("action"+lab[0])
                self.update_action_labels(lab[0])

                previous_status = self.last_status
                current_status = self.fatigue_detector.detec_fatigue(eyear, mouthar)
            else:
                current_status = 'normal'
                self.fatigue_detector.eye_cycle_count = 0
                self.fatigue_detector.mouth_cycle_count = 0
                print('未检测到人脸，重置状态')
            self.last_status = current_status

            #### 状态更新
            if current_status!=previous_status:
                self.update_fatigue_status(current_status)
                if current_status == 'fatigue' and not self.warning_playing:
                    #self.warning_playing = True
                    Thread(target=self._play_warning).start()
                elif current_status != 'fatigue':
                    self.warning_playing = False


            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 等比例缩放
            label_size = self.parent_window.label_20.size()
            h, w = rgb_frame.shape[:2]

            target_w = label_size.width()
            target_h = int(h*(target_w/w))

            if target_h > label_size.height():
                target_h = label_size.height()
                target_w = int(w * (target_h / h))

            resized = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # 显示图像
            q_img = QImage(
                resized.data,
                target_w,
                target_h,
                resized.strides[0],
                QImage.Format_RGB888
            )

            self.parent_window.label_20.setPixmap(QPixmap.fromImage(q_img))
            self._update_statistics()
        except Exception as e:
            print(e)
            self.cleanup()

    def _update_statistics(self):
        detector = self.fatigue_detector

        self.parent_window.label_9.setText(
            f'<html><head><body><p align="center">'
            f'<span style="font-size:10pt;font-weight:600;">'
            f'{detector.total_eye_closed}</span></p></body></html>'
        )
        self.parent_window.label_11.setText(
            f'<html><head><body><p align="center">'
            f'<span style="font-size:10pt;font-weight:600;">'
            f'{detector.total_mouth_open}</span></p></body></html>'
        )

        self._update_thirty_sec_status()

    def _update_thirty_sec_status(self):

        detector = self.fatigue_detector

        self.parent_window.label_10.setText(
            f'<html><head><body><p align="center">'
            f'<span style="font-size:10pt;font-weight:600;">'
            f'{detector.thirty_sec_eye}</span></p></body></html>'
        )

        self.parent_window.label_12.setText(
            f'<html><head><body><p align="center">'
            f'<span style="font-size:10pt;font-weight:600;">'
            f'{detector.thirty_sec_mouth}</span></p></body></html>'
        )

    # 释放摄像头
    def cleanup(self):
        if self.cleaned:
            return
        print("释放摄像头资源...")

        try:
            if hasattr(self, 'timer'):
                self.timer.timeout.disconnect()
                self.timer.stop()

            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            print(e)
        self.cleaned = True

# 主窗口
class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.init_ui()
        self.camera = None

    # 初始化UI
    def init_ui(self):
        self.pushButton.clicked.connect(self.reset_counters)
        self.init_camera()

    def reset_counters(self):
        counters = [
            (self.label_10, 0),
            (self.label_12, 0),
            (self.label_13, 0),
        ]

        for label, value in counters:
            label.setText(
                f"<html><head/><body><p align='center'>"
                f"<span style='font-size:10pt;font-weight:600;'>"
                f"{value}</span></p ></body></html>"
            )
        # self.camera.update_fatigue_status('normal')
        # if hasattr(self, 'camera') and self.camera:
        #     self.camera.fatigue_detector.reset_counters()


    def init_camera(self, retry_count=3):
        for i in range(retry_count):
            try:
                self.camera = CameraController(self)
                return
            except Exception as e:
                if i == retry_count-1:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "错误",
                        f'摄像头初始化失败, {e}'
                    )
                else:
                    print(f"初始化失败, 正在重试...{i+1}次")
                    QtCore.QThread.nsleep(1000)

    def closeEvent(self, event):
        if hasattr(self, 'camera'):
            self.camera.cleanup()
        super().closeEvent(event)

    def update_counters(self):
        try:
            counters = [
                (self.label_10, config.TOTAL),
                (self.label_12, config.mTOTAL),
                (self.label_13, config.hTOTAL),
            ]

            for label, value in counters:
                label.setText(
                    f"<html><head/><body><p align='center'>"
                    f"<span style='font-size:10pt;font-weight:600;'>"
                    f"{value}</span></p ></body></html>"
                )
        except Exception as e:
            print(f'计数器更新失败: {e}')

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        return app.exec()
    except Exception as e:
        print(e)
        return 1
    finally:
        print("应用程序退出")

if __name__ == '__main__':
    sys.exit(main())