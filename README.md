# Yolo-V5s转换ncnn并推理测试Pipeline


## 步骤

### 1、yolov5代码修改：

    common.py中的class Focus(nn.Module)中，切片操作改为下采样<torch.nn.functional.interpolate>
    
    要修改的common.py就在本文件夹下，可以替换到yolov5/models下面
    
    修改后，加载原来的模型再训一遍，否则结果不能看。
    
    
### 2、yolov5模型转换成onnx：

   参考[官方链接](https://github.com/ultralytics/yolov5/issues/251)
```   
    python models/export.py --weights yolov5s.pt --img 640 --batch 1  # export at 640x640 with batch size 1
```   
    (注意代码中设置 model.model[-1].export = True)
    
    
### 3、onnx模型简化：
```   
    pip3 install onnx-simplifier
    python3 -m onnxsim yolov5s.onnx yolov5s.onnx
```   
    
    
### 4、编译安装ncnn：

   参考[官方链接](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)
    
    注意其中的依赖项，尤其是protobuf，亲测2.5.0不行，需要源码升级为3.0以上，注意源码升级时下载cpp版即可
```   
    make && make install
```   
    
    
### 5、在ncnn中转换onnx为bin/param：
    
    ncnn安装后，在 ncnn/build/tools/onnx 下面 找到 onnx2ncnn
    
    执行下面命令即可：
```
    ./onnx2ncnn yolov5s.onnx yolov5s.param yolov5s.bin
```


### 6、C++推理代码:

   推理代码参考(自己的github)[https://github.com/a954217436/yolov5s_ncnn_inference.git]
    
    输出层名字可以使用netron查看，我查到的是output、857、877，不同模型可能不一样
    
    注意CMakeLists.txt中，需要openMP，openCV，libncnn.a等
    
```
    mkdir build && cd build
    cmake ..
    make
    ./yolov5.test ../test_jpg
```


### TODO:
    
* vulkan 推理测试
* 5m/5l/5x 转换
    

