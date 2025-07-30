# 调用dlib库检测人脸特征点

from imutils import face_utils
import  numpy as np
import  dlib
import cv2
import scipy.spatial.distance as dist
# 获取面部特征索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 初始化DLIB的人脸检测器和面部标志点预测器
print("[INFO] loading facial landmark predictor...")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")


# 检测人脸特征点，并且绘制特征点
def  draw_facial_features(frame,shape):
    # 绘制眼睛轮廓
    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]
    for eye in [left_eye, right_eye]:
        hull = cv2.convexHull(eye)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

    # 绘制嘴巴的轮廓
    mouth = shape[mStart:mEnd]
    mouth_hull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

    #绘制纵横线
    for (start,end) in [(38,40),(43,47),(51,57),(48,54)]:
        cv2.line(frame,tuple(shape[start]),tuple(shape[end]),(0,255,0),1)


    return frame


# 检测人脸
def detect_fatigue(frame):
    # 图像的灰度化
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 检测人脸信息
    rects = detector(gray,0)
    shape = None

    # 从检测信息中提取人脸的特征点
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        # 计算两只眼睛的比例
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        eyear = (calculate_eye_aspect_ratio(left_eye)+calculate_eye_aspect_ratio(right_eye))/2.0

        # 计算嘴巴的比例
        mouth = shape[mStart:mEnd]
        mouthar = calculate_mouth_aspect_ratio(mouth)


    frame = draw_facial_features(frame,shape)
    return frame,eyear,mouthar


#计算眼睛的EAR
def calculate_eye_aspect_ratio(eye):
    # 垂直距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 水平距离
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

#计算嘴巴纵横比(MAR)
def calculate_mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    return (A + B) / (2.0 * C)