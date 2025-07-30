import sys
import  atexit
import cv2
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from Window import Ui_Form
from config import config

# 实现窗口
class MainWindow(QtWidgets.QWidget,Ui_Form):
    # 初始化
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.init_ui()
        self.camera = None

    #初始化ui
    def init_ui(self):
        self.pushButton.clicked.connect(self.update_counters)
        self.init_camera()

    # 清空计数器
    def update_counters(self):
        try:
            #三个标签
            counters =[
                (self.label_9,config.TOTAL),
                (self.label_11,config.mTOTAL),
                (self.label_13,config.hTOTAL)
            ]

            for label,value in counters:
                label.setText(
                    f"<html><head/><body><p align='center'>"
                    f"<span style='font-size:10pt;font-weight:600;'>"
                    f"{value}</span></p ></body></html>"
                )

        except Exception as e:
            print("更新计数器失败！")

    # 初始化摄像头
    def init_camera(self,retry_count=3):
        for i in range(retry_count):
            try:
                self.camera = CameraController(self)
            except Exception as e:
                if i == retry_count - 1:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "错误",
                        f"摄像头初始化失败: {str(e)}\n请检查设备连接"
                    )
                else:
                    print(f"初始化失败，正在重试...({i + 1}/{retry_count})")
                    QtCore.QThread.msleep(1000)


    # 释放资源
    def closeEvent(self,event):
        if hasattr(self,'camera'):
            self.camera.cleanup()
        super().closeEvent(event)



# 实现摄像头
class CameraController():
    # 初始化
    def __init__(self,parent_window):
        self.parent = parent_window
        self.timer = QTimer()
        self._cleaned = False

        # 三重释放保障
        self.parent.destroyed.connect(self.cleanup)
        atexit.register(self.cleanup)

        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("摄像头设备未就绪")

            self.timer.timeout.connect(self.update_frame)
            self.timer.start(20)  # 50 FPS

        except Exception as e:
            self.cleanup()
            raise

    # 更新摄像头的帧
    def update_frame(self):
        try:
            ret,frame = self.cap.read()
            if not ret:
                return
            # 图像处理流水线
            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 智能缩放
            label_size = self.parent.label_20.size()
            h, w = rgb_image.shape[:2]
            target_w = label_size.width()
            target_h = int(h * (target_w / w))

            if target_h > label_size.height():
                target_h = label_size.height()
                target_w = int(w * (target_h / h))

            resized = cv2.resize(rgb_image, (target_w, target_h),
                                 interpolation=cv2.INTER_AREA)

            # 显示图像
            q_img = QImage(
                resized.data,
                target_w,
                target_h,
                resized.strides[0],
                QImage.Format_RGB888
            )
            self.parent.label_20.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print("出现帧错误!")
            self.cleanup()
            raise

    # 释放资源
    def cleanup(self):
        if self._cleaned:
            return
        print("释放摄像头资源")
        try:
            if hasattr(self,'timer'):
                self.timer.timeout.disconnect()
                self.timer.stop()
            if hasattr(self,'cap') and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            print(f"资源释放异常：{e}")
        self._cleaned = True



#调用窗口、摄像头
def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception as e:
        print(f"致命错误:{e}")
        return 1
    finally:
        print("应用程序退出！")

if __name__ == '__main__':
    sys.exit(main())