## 1st solution - face recognition challenge
- https://www.aivivn.com/contests/2 

Vì mình biết rất nhiều công ty và nhiều bạn đang làm hoặc nghiên cứu về bài toán này nên mình sẽ cố viết chi tiết và dễ hiểu nhất có thể để mọi người sử dụng. Source code này dùng cho cuộc thi, nếu chạy cho product thì chỉ cần dùng model tốt nhất trong 14 model là có thể đạt accuracy hơn 94%. Mình chưa làm face recognition bao giờ nên 10 ngày quá ít để có thể đạt độ chính xác cao như mong muốn.
### Environments
- Ubuntu 16.04
- Cuda 9.0
- Cudnn 7
- OpenCV 3
- Both python 3.5 and 2.7
- Pytorch, keras
- Caffe [install https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/]
- [2D-3D Face Alignment] https://github.com/1adrianb/face-alignment
- [Dlib 19.17] https://github.com/davisking/dlib/releases
    - Với dlib-19.17, chạy lệnh sau để cài đặt (ko dùng pip vì pip sẽ cài mặc định bản 19.15 xuất hiện nhiều lỗi):
    - $ python3 setup.py install
- [Insightface] https://github.com/deepinsight/insightface
- [Facenet] https://github.com/davidsandberg/facenet
- [VGG Face2] https://github.com/ox-vgg/vgg_face2

### Step1: Face detection and alignment
- Tải datasets trên trang aivivn và bỏ vào thư mục datasets
- Chạy file align_face.py trong thư mục mysrc
    - $ cd mysrc
    - $ python3 align_face.py
- Output:
    - thư mục datasets/aligned/ chứa ảnh đã được align dùng cho lấy embedding và thư mục datasets/unknown chứa những ảnh ko detect được face
- Key:
    - trong tập dataset có rất nhiều ảnh mặt quay chỉ thấy một phía và nhiều ảnh đầu bị chúc xuống nên MTCNN ko phát hiện được khuôn mặt hoặc phát hiện được (hạ threshold thấp xuống) nhưng predict tọa độ 5 điểm sai khiến align ko chuẩn (fail 60 ảnh trong train và hơn 200 ảnh trong test), single shot scale-invariant face detector trong thư viện 2D-3D Face Alignment giải quyết tốt hơn nhiều so với MTCNN (fail 0 ảnh train và 24 ảnh test, hầu hết 24 ảnh này ko có mặt hoặc chỉ là tranh vẽ) - xem trong thư mục datasets/unknown/test
    - một số ảnh bị răng cưa nên 2D-3D Face Alignment cũng fail nên mình blur ảnh thì nó lại detect đươc
### Step2: Split dataset
- Chia dataset thảnh 5 fold, chaỵ 30 lần với các random state khác nhau
    - $ cd mysrc
    - $ python3 split_dataset.py
- Output: 2 file datasets/train_refined.csv và datasets/test_refined.csv
### Step3: Extract embedding
    models
        ├── dlib-19.17
        │   ├── models
        │       ├── dlib_face_recognition_resnet_model_v1.dat
        │       └── shape_predictor_5_face_landmarks.dat
        ├── facenet
        │   ├── models
        │       ├── 20180402-114759
        │       │   ├── 20180402-114759.pb
        │       │   ├── model-20180402-114759.ckpt-275.data-00000-of-00001
        │       │   ├── model-20180402-114759.ckpt-275.index
        │       │   └── model-20180402-114759.meta
        │       └── 20180408-102900
        │           ├── 20180408-102900.pb
        │           ├── model-20180408-102900.ckpt-90.data-00000-of-00001
        │           ├── model-20180408-102900.ckpt-90.index
        │           └── model-20180408-102900.meta
        ├── insightface
        │   ├── models
        │       ├── model-r100-ii
        │       │   ├── model-0000.params
        │       │   └── model-symbol.json
        │       ├── model-r34-amf
        │       │   ├── model-0000.params
        │       │   └── model-symbol.json
        │       ├── model-r50-am-lfw
        │       │   ├── model-0000.params
        │       │   └── model-symbol.json
        │       └── model-y1-test2
        │           ├── model-0000.params
        │           └── model-symbol.json
        └── vgg_face2
            ├── resnet50_128_pytorch
            │   ├── resnet50_128_pytorch.pth
            │   └── resnet50_128_pytorch.py
            ├── resnet50_256_pytorch
            │   ├── resnet50_256_pytorch.pth
            │   └── resnet50_256_pytorch.py
            ├── resnet50_ft_pytorch
            │   ├── resnet50_ft_pytorch.pth
            │   └── resnet50_ft_pytorch.py
            ├── resnet50_scratch_pytorch
            │   ├── resnet50_scratch_pytorch.pth
            │   └── resnet50_scratch_pytorch.py
            ├── senet50_128_pytorch
            │   ├── senet50_128_pytorch.pth
            │   └── senet50_128_pytorch.py
            ├── senet50_256_pytorch
            │   ├── senet50_256_pytorch.pth
            │   └── senet50_256_pytorch.py
            ├── senet50_ft_pytorch
            │   ├── senet50_ft_pytorch.pth
            │   └── senet50_ft_pytorch.py
            └── senet50_scratch_pytorch
                ├── senet50_scratch_pytorch.pth
                └── senet50_scratch_pytorch.py

- Tải pretrain model của 4 thư viện ở mục environments và đặt vào các thư mục theo đúng đường dẫn như cây thư mục trên:
- Chạy 4 script sau để trích xuất embedding của 4 model:
    - $ cd mysrc
    - $ bash generate_insightface_embedding.sh
    - $ bash generate_facenet_embedding.sh
    - $ bash generate_dlib_embedding.sh
    - $ bash generate_vggface2_embedding.sh
- Chú ý vggface2 phải cài caffe và chạy trên python2.7
- Output: thư mục embedding trong các model[models/dlib-19.17/embedding models/insightface/embedding models/vgg_face2/embedding models/facenet/embedding] chứa các vector embedding tương ứng các ảnh của dataset.
- Key: rất nhiều ảnh của dataset là ảnh grayscale, và bị răng cưa (ảnh độ phân giải thấp resize lên phân giải cao), những ảnh này có thể xếp vào trường hợp khó nhận dạng. Lúc đầu mình sử dụng 2 thư viện [colorization] https://github.com/richzhang/colorization để tạo màu cho ảnh và [Super Resolution GANs] https://github.com/tensorlayer/srgan để resize về resolution[224x224] để làm đầu vào cho vgg face2, giúp mình boost accuracy lên khoảng 0.004. Nhưng sau đó mình chọn cách dùng augmentation trong lúc train, nhiều phép blur, tograyscale, contrast, addToHueAndSaturation ... nên không cần sử dụng thêm 2 thư viện này. Top1 accuracy có thể improve khoảng 0.007.

### Step4: Train on trainset(4720 images)
- Note: model mình build chỉ khoảng 2M parameters nên chỉ cần GPU khoảng 3GB là đủ train, vì mình load tất cả ảnh vào RAM để train cho nhanh nên để chạy code thành công yêu cầu RAM > 24GB, nếu ko đủ RAM các bạn có thể sửa lại hàm train_generator để train với batchsize nhỏ hơn nhưng lâu hơn 1 chút. Mình có sử dụng augmentation và cyclical learning rates cho train và flip augmetation cho test.
- Có 15 models (thực chất chỉ dùng 14 models, dlib yếu quá nên mình ko dùng vì sợ nhiễu), mỗi model sẽ tương ứng với một pretrain weights xem ở cuối bài
- cd vào mỗi model và chaỵ 2 lệnh
    - $ python3 prepare.py
    - $ python3 train.py
- Mỗi model mình sẽ chạy 4 lần với các random state khác nhau khi split trainset and validset, sau khi train xong 15 models, sẽ tạo ra 15 file ptest.npy và checkpoints trong thư mục weights, sau đó chạy:
    - $ python3 generate_pseudo.py
- Output: file datasets/pseudo_test.csv chứa label và prob tương ứng mỗi ảnh
- Hoặc có thể chạy script sẵn trong thư mục mysrc:
    - $ bash prepare_and_train_step4.sh

### Step5: Add pseudo dataset and predict unknown images
- Key: Cái này mình đã từng áp dụng cho cuộc thi Zalo AI challenge và đạt TOP1 nên cuộc thi này tiếp tục xài, mình tạo file submission chỉ có dự đoán 1 class, 4 class còn lại mình để giá trị -1, submit lên đạt 0.936 trên bộ public, giả sử public và private có cùng distribution và ko shakeup quá nhiều, có nghĩa dự đoán sai (1-0.936)*17091 = 1093 ảnh, mình sẽ chọn những ảnh có prob > 0.65 trong tập test (trong file datasets/pseudo_test.csv) đưa vào train model, lúc này bộ trainset sẽ có thêm 14300 images, khoảng 17091-14300 =  2791 ảnh mình đề phòng unknown và dự đoán sai. Đương nhiên trong 14300 ảnh này sẽ có dự đoán sai nhưng quá ít nên model có khả năng bỏ qua những trường hợp này.
- Xác định unknown:
    - Trước đây mình có kinh nghiệm dùng gans trong anomaly detection, ví dụ tìm bất thường trong vật thể mà tập train chỉ có ảnh bình thường (ko có ảnh bất thường), ví dụ các bài toán tìm chi tiết máy lỗi, gãy xương, xác định ung thư có thể giải quyết với độ chính xác trên 90% nên khá tự tin detect được unknown. Tiếc là mình thử và nó không hoạt động trên bài toán này, có thể nó chỉ tốt với vài toán chi tiết nhỏ nhưng phải xác định tổng quan cả khuôn mặt như face thì fail. Bạn có thể dùng giải thuật EVM hoặc OpenMax có thể giải quyết được bài toán detect unknown này, mình chưa thử.
    - Mình quay lại với cách cổ điển là dùng euclidean distance giữa các embedding và kết hợp dùng prob trong lúc predict, không đạt độ chính xác cao nhưng tầm trên 50% là đủ chiến thắng cuộc thi này.
    - Chạy các lệnh sau để tính embedding distance giữa ảnh test và tập train:
        - $ cd models/embedding_distance
        - $ bash insightface_embedding_distance.sh
    - Mình chỉ dùng embedding của insightface để xác định unknown, có thể kết hợp các model khác của facenet, vggface2, dlib nếu ai thích, mình có viết sẵn script trong thư mục này.
    - Mình tạo 1 file submission predict tất cả là nhãn 1000, score là 0.01 có nghĩa xấp xỉ 17000*0.01=170 tấm unknown trong tập public test, nếu private và public ko shakeup và cùng distribution thì xấp xỉ 340 unknown images trong toàn bộ tập test.
    - Mình quy ước những ảnh trong tập test có euclidean distance minimum > 1.35 và prob predicted < 0.1 là unknown, có thể giảm ngưõng distance tăng ngưỡng prob nếu muốn nhưng mình muốn lấy unknown chính xác nhất có thể để đưa vào bộ train nên chọn 2 ngưỡng này. Kết hợp ảnh ko detect được face ở step1 để tạo unknown class. Mình sẽ có tầm 116 tấm làm ground truth cho unknown, đưa vào train thôi.
    - Chạy lệnh:
    - $ python3 get_unknown.py
    - Output: file datasets/pseudo_train.csv chứa pseudo test và unknown class

### Step6: Train on trainset + pseudo testset
- Chạy 2 lệnh trong 15 models(có thể bỏ qua dlib nếu muốn):
    - $ python3 prepare_pseudo.py
    - $ python3 train_pseudo.py
- Sau khi train xong chạy lệnh để tạo file submission:
    - $ cd mysrc
    - $ python3 pseudo_generate_submission_step6.py
    - Output: file mysrc/submission_pseudo_step2.csv để submit test thử và file datasets/pseudo_train_step2.csv train step 7
### Step7: Continue train on trainset + pseudo testset
- Chạy 2 lệnh trong 15 models(có thể bỏ qua dlib nếu muốn):
    - $ python3 prepare_pseudo_step2.py
    - $ python3 train_pseudo_step2.py
- Sau khi train xong chạy lệnh để tạo file submission:
    - $ cd mysrc
    - $ python3 pseudo_generate_submission_step7.py
    - Output: file mysrc/submission_final.csv
    - Bạn có thể lặp lại bước 7 nếu muốn, có thể acc sẽ tăng nhưng nguy cơ overfit cũng tăng, đến đây mình dừng lại.

- Mình có thống kê top1 accuracy trên bộ valid cho 15 model sau qua các bước 4,6,7 để các bạn có 1 cái nhìn tổng quan (xem file training_log*.txt trong mỗi model)

    Model                     | Pretrained weights       | Valid acc step4 (1000 classes) | Valid acc step 6 (1001 classes) | Valid acc step7 (1001 classes)
    -------------------------:| :-----------------------:|:------------------------------ |:-------------------------------:|:------------------------------:
    model1_insightface        | model-r100-ii            | 0.887447                       | 0.940572                        | 0.945180
    model2_insightface        | model-r34-amf            | 0.847458                       | 0.925953                        | 0.930456
    model3_insightface        | model-r50-am-lfw         | 0.855244                       | 0.929502                        | 0.933686
    model4_insightface        | model-y1-test2           | 0.787871                       | 0.905667                        | 0.912235
    model5_dlib               | dlib_resnet_model_v1.dat | 0.341261                       |                                 | 0.619703
    model6_facenet            | 20180402_114759          | 0.670604                       | 0.857786                        | 0.862553
    model7_facenet            | 20180408_102900          | 0.682468                       | 0.872087                        | 0.879078
    vggface2_resnet50_128     | resnet50_128_pytorch     | 0.752331                       | 0.893167                        | 0.898305
    vggface2_resnet50_256     | resnet50_256_pytorch     | 0.756780                       | 0.902648                        | 0.905720
    vggface2_resnet50_ft      | resnet50_ft_pytorch      | 0.779237                       | 0.910328                        | 0.916525
    vggface2_resnet50_scratch | resnet50_scratch_pytorch | 0.756515                       | 0.889831                        | 0.910328
    vggface2_senet50_128      | senet50_128_pytorch      | 0.753178                       | 0.892373                        | 0.895763
    vggface2_senet50_256      | senet50_256_pytorch      | 0.757998                       | 0.901536                        | 0.907044
    vggface2_senet50_ft       | senet50_ft_pytorch       | 0.773464                       | 0.908422                        | 0.913824
    vggface2_senet50_scratch  | senet50_scratch_pytorch  | 0.747087                       | 0.901059                        | 0.906992

- File submission cuối cùng là tổng hợp của 14 model theo tỷ lệ thích hợp, model có valid accuracy cao sẽ đánh trọng số cao [xem ở line 45 file pseudo_generate_submission_step7.py]

Nếu bạn thấy hữu ích thì connect linkedin và endorse một số skill của mình ^^ https://www.linkedin.com/in/dung-nguyen-ba-137027159/ .