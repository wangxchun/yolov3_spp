# YOLOv3 SPP
## 該項目源自[ultralytics/yolov3](https://github.com/ultralytics/yolov3)
## 1 環境配置：
* Python3.6或者3.7
* Pytorch1.7.1(注意：必須是1.6.0或以上，因為使用官方提供的混合精度訓練1.6.0後才支持)
* pycocotools(Linux: `pip install pycocotools`;   
  Windows: `pip install pycocotools-windows`(不需要額外安裝vs))
* 更多環境配置信息，請查看`requirements.txt`文件
* 最好使用GPU訓練

## 2 文件結構：
```
  ├── cfg: 配置文件目錄
  │    ├── hyp.yaml: 訓練網絡的相關超參數
  │    └── yolov3-spp.cfg: yolov3-spp網絡結構配置 
  │ 
  ├── data: 存儲訓練時數據集相關信息緩存
  │    └── pascal_voc_classes.json: pascal voc數據集標籤
  │ 
  ├── runs: 保存訓練過程中生成的所有tensorboard相關文件
  ├── build_utils: 搭建訓練網絡時使用到的工具
  │     ├── datasets.py: 數據讀取以及預處理方法
  │     ├── img_utils.py: 部分圖像處理方法
  │     ├── layers.py: 實現的一些基礎層結構
  │     ├── parse_config.py: 解析yolov3-spp.cfg文件
  │     ├── torch_utils.py: 使用pytorch實現的一些工具
  │     └── utils.py: 訓練網絡過程中使用到的一些方法
  │
  ├── train_utils: 訓練驗證網絡時使用到的工具(包括多GPU訓練以及使用cocotools)
  ├── weights: 所有相關預訓練權重(下面會給出百度雲的下載地址)
  ├── model.py: 模型搭建文件
  ├── train.py: 針對單GPU或者CPU的用戶使用
  ├── train_multi_GPU.py: 針對使用多GPU的用戶使用
  ├── trans_voc2yolo.py: 將voc數據集標注信息(.xml)轉為yolo標注格式(.txt)
  ├── calculate_dataset.py: 1)統計訓練集和驗證集的數據並生成相應.txt文件
  │                         2)創建data.data文件
  │                         3)根據yolov3-spp.cfg結合數據集類別數創建my_yolov3.cfg文件
  └── predict_test.py: 簡易的預測腳本，使用訓練好的權重進行預測測試
```

## 3 訓練數據的準備以及目錄結構
* 這裡建議標注數據時直接生成yolo格式的標籤文件`.txt`，推薦使用免費開源的標注軟件(支持yolo格式)，[https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)
* 如果之前已經標注成pascal voc的`.xml`格式了也沒關係，我寫了個voc轉yolo格式的轉化腳本，4.1會講怎麼使用
* 測試圖像時最好將圖像縮放到32的倍數
* 標注好的數據集請按照以下目錄結構進行擺放:
```
├── my_yolo_dataset 自定義數據集根目錄
│         ├── train   訓練集目錄
│         │     ├── images  訓練集圖像目錄
│         │     └── labels  訓練集標籤目錄 
│         └── val    驗證集目錄
│               ├── images  驗證集圖像目錄
│               └── labels  驗證集標籤目錄            
```

## 4 利用標注好的數據集生成一系列相關準備文件，為了方便我寫了個腳本，通過腳本可直接生成。也可參考原作者的[教程](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)
```
├── data 利用數據集生成的一系列相關準備文件目錄
│    ├── my_train_data.txt:  該文件里存儲的是所有訓練圖片的路徑地址
│    ├── my_val_data.txt:  該文件里存儲的是所有驗證圖片的路徑地址
│    ├── my_data_label.names:  該文件里存儲的是所有類別的名稱，一個類別對應一行(這裡會根據`.json`文件自動生成)
│    └── my_data.data:  該文件里記錄的是類別數類別信息、train以及valid對應的txt文件
```

### 4.1 將VOC標注數據轉為YOLO標注數據(如果你的數據已經是YOLO格式了，可跳過該步驟)
* 使用`trans_voc2yolo.py`腳本進行轉換，並在`./data/`文件夾下生成`my_data_label.names`標籤文件，
* 執行腳本前，需要根據自己的路徑修改以下參數
```python
# voc數據集根目錄以及版本
voc_root = "./VOCdevkit"
voc_version = "VOC2012"

# 轉換的訓練集以及驗證集對應txt文件，對應VOCdevkit/VOC2012/ImageSets/Main文件夾下的txt文件
train_txt = "train.txt"
val_txt = "val.txt"

# 轉換後的文件保存目錄
save_file_root = "/home/wz/my_project/my_yolo_dataset"

# label標籤對應json文件
label_json_path = './data/pascal_voc_classes.json'
```
* 生成的`my_data_label.names`標籤文件格式如下
```text
aeroplane
bicycle
bird
boat
bottle
bus
...
```

### 4.2 根據擺放好的數據集信息生成一系列相關準備文件
* 使用`calculate_dataset.py`腳本生成`my_train_data.txt`文件、`my_val_data.txt`文件以及`my_data.data`文件，並生成新的`my_yolov3.cfg`文件
* 執行腳本前，需要根據自己的路徑修改以下參數
```python
# 訓練集的labels目錄路徑
train_annotation_dir = "/home/wz/my_project/my_yolo_dataset/train/labels"
# 驗證集的labels目錄路徑
val_annotation_dir = "/home/wz/my_project/my_yolo_dataset/val/labels"
# 上一步生成的my_data_label.names文件路徑(如果沒有該文件，可以自己手動編輯一個txt文檔，然後重命名為.names格式即可)
classes_label = "./data/my_data_label.names"
# 原始yolov3-spp.cfg網絡結構配置文件
cfg_path = "./cfg/yolov3-spp.cfg"
```

## 5 預訓練權重下載地址（下載後放入weights文件夾中）：
* `yolov3-spp-ultralytics-416.pt`: 鏈接: 
* `yolov3-spp-ultralytics-512.pt`: 鏈接: 
* `yolov3-spp-ultralytics-608.pt`: 鏈接:
* `yolov3spp-voc-512.pt`: 鏈接: 
 
 
## 6 數據集，本例程使用的是PASCAL VOC2012數據集
* `Pascal VOC2012` train/val數據集下載地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

## 7 使用方法
* 確保提前準備好數據集
* 確保提前下載好對應預訓練模型權重
* 若要使用單GPU訓練或者使用CPU訓練，直接使用train.py訓練腳本
* 若要使用多GPU訓練，使用`python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`指令,`nproc_per_node`參數為使用GPU數量
* 訓練過程中保存的`results.txt`是每個epoch在驗證集上的COCO指標，前12個值是COCO指標，後面兩個值是訓練平均損失以及學習率
