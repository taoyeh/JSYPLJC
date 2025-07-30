# 实现帧的检测，帧的体现
# 文件的进口是原始的帧
# 文件的出口处理后的帧
import  time
import  cv2
import  myfatigue
import  mydetect
_fps_state = {
    'frame_count': 0,
    'start_time': time.time(),
    'last_update': time.time(),
    'fps': 0.0
}
def frametest(frame):
    # 返回检测到的结果
    ret = []
    labellist= []
    if frame is None or not hasattr(frame,'shape'):
        return frame
    try:
        _fps_state['frame_count']+=1
        current_time = time.time()
        # 调用dlib库检测人脸并绘制人脸的轮廓
        frame,eyear,mouthar= myfatigue.detect_fatigue(frame)
        labellist,frame= detect_action(frame,labellist)


        #print("ear:"+str(eyear)+"mar"+str(mouthar))
        ret.append(labellist)
        ret.append(round(eyear,3))
        ret.append(round(mouthar, 3))




        # 每秒更新FPS
        if current_time - _fps_state['last_update'] >= 1.0:
            _fps_state['fps'] = _fps_state['frame_count'] / (
                    current_time - _fps_state['start_time'])  # 计算FPS = 帧数 / 时间间隔
            _fps_state['frame_count'] = 0  # 重置帧计数器
            _fps_state['start_time'] = current_time  # 更新开始时间和最后更新时间戳
            _fps_state['last_update'] = current_time
        # 确保帧可写（处理UMat等特殊情况）检查帧是否可写（某些OpenCV操作会产生只读或特殊格式的帧）
        if not getattr(frame, 'flags', None) or not frame.flags.writeable:
            frame = frame.copy()
            # 添加FPS显示（BGR格式红色文字）
        cv2.putText(
            img=frame,
            text=f"FPS:{_fps_state['fps']:.1f}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        return frame,ret

    except Exception as e:
        return  frame,ret


def detect_action(frame,labellist):
    action=mydetect.predict(frame)
    for label, prob, xyxy in action:
        # 在labellist加入当前label
        labellist.append(label)
        # 将标签和置信度何在一起
        text = label + str(prob)
        # 画出识别框
        left = int(xyxy[0])
        top = int(xyxy[1])
        right = int(xyxy[2])
        bottom = int(xyxy[3])
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        # 在框的左上角画出标签和置信度
        cv2.putText(frame, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
    return labellist, frame



