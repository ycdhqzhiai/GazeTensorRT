# GazeTensorRT
gaze estimation tensorrt

## run
* 生成onnx模型和wts文件
```shell
git clone https://github.com/ycdhqzhiai/Gaze-PFLD
cd Gaze-PFLD
python genwts.py
#同时生成onnx方便查看网络结构
python export_onnx.py
```
* 编译运行
```shell
mkdir build
cd build
cmake ..
make -j4
#1.得到engin
./gaze-pfld -s
#2 推理运行
./gaze-pfld -d
```
* 耗时

| 框架(GTX3090)                             | inference(ms)  |
| ------------------------------------ | ----- |
| tensorrt                              | 1.1 |
| torch                              | 13.7 |
| onnxruntime | 3.8~270(不知道为什么) |

* 关于误差</br>
对pth onnx engine三个模型都进行了测试，gaze部分，pth和engine基本一致，误差在小数点后第三位，但是landmarks部分误差较大，在小数点后第二位，onnx与engine也有一定误差，基本也在小数点后第二位，并且很奇怪的是，pth模型和onnx模型推理的结果也有一定误差，在小数点后第二位，令人不解

## Reference resources
* 1.https://github.com/wang-xinyu/tensorrtx</br>
* 2.https://github.com/ycdhqzhiai/Gaze-PFLD
