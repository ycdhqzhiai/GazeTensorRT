import cv2
import numpy as np
import onnxruntime

session_onnx = onnxruntime.InferenceSession('gaze-pfld.onnx')

batch_size = session_onnx.get_inputs()[0].shape[0]
img_size_h = session_onnx.get_inputs()[0].shape[2]
img_size_w = session_onnx.get_inputs()[0].shape[3]
input_name = session_onnx.get_inputs()[0].name


def preprocess(img):
    img = cv2.resize(img, (img_size_w, img_size_h))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_in = img / 256.0
    img_in = np.expand_dims(img_in, axis=0).astype(np.float32)
    return img_in


img = cv2.imread('pre_img.png')

img_in = preprocess(img)
#print(img_in)
landmarks, gaze = session_onnx.run(None, {input_name: img_in})
print(landmarks)
landmarks = landmarks.reshape(-1,2)
for i in range(landmarks.shape[0]):
    x_y = landmarks[i]
    cv2.circle(img, (int(x_y[0] * img.shape[1]), int(x_y[1] * img.shape[0])), 1, (0, 0, 255),-1)

eye_c = landmarks[-1]
cv2.line(img, (int(eye_c[0]*img.shape[1]), int(eye_c[1]*img.shape[0])), (int(eye_c[0]*img.shape[1] + gaze[0][0]*200), int(eye_c[1]*img.shape[0] + gaze[0][1]*200)), (0,255,255), 2)
cv2.imshow('result', img)
cv2.waitKey(0)
