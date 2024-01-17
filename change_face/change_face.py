import cv2
import dlib
import numpy as np
import ffmpeg

# 初始化人脸检测器
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 请下载相应的预训练模型
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # 请下载相应的预训练模型

# 加载输入照片1、照片2和照片3
image1 = cv2.imread("photo1.jpg")
image2 = cv2.imread("photo2.jpg")
image3 = cv2.imread("photo3.jpg")

# 获取人脸特征点
def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) == 0:
        return None
    shape = shape_predictor(image, faces[0])
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
    return landmarks

landmarks1 = get_face_landmarks(image1)
landmarks2 = get_face_landmarks(image2)
landmarks3 = get_face_landmarks(image3)

# 创建输出视频
output_video = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (image1.shape[1], image1.shape[0]))

# 逐帧生成混合视频
for alpha in np.linspace(0, 1, 100):  # 生成100帧
    blended_image = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
    output_video.write(np.uint8(blended_image))

for alpha in np.linspace(0, 1, 100):  # 生成100帧
    blended_image = cv2.addWeighted(image2, 1 - alpha, image3, alpha, 0)
    output_video.write(np.uint8(blended_image))

# 释放视频写入器
output_video.release()

# 使用ffmpeg将AVI转换为MP4
ffmpeg.input("output_video.avi").output("output_video.mp4").run()
