The MODEL5 in FINAL_S7.ipynb refers to MODEL5 in models.py , and same restructured to MODEL6 in model_3.py
# Findings

Speaking of the Final Model
The usage of droput to 0.05 really helped
and reducing the batch size to 64 helped me in achieving the accuarcy of 99.4 twice
addition of ReLU immidiate after convolution became one of the plus point, instead of adding it after batch normalization

True
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160        ---> To extact base features as much as possible
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 12, 24, 24]           1,740        ---> To hit a 5X5 receptive field
              ReLU-6           [-1, 12, 24, 24]               0
       BatchNorm2d-7           [-1, 12, 24, 24]              24
           Dropout-8           [-1, 12, 24, 24]               0
            Conv2d-9           [-1, 24, 24, 24]             312        ---> to learn some more patterns
        MaxPool2d-10           [-1, 24, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]           2,604        ---> to learn more patterns
             ReLU-12           [-1, 12, 10, 10]               0
      BatchNorm2d-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 14, 8, 8]           1,526        ---> to learn  more patterns
             ReLU-16             [-1, 14, 8, 8]               0
      BatchNorm2d-17             [-1, 14, 8, 8]              28
          Dropout-18             [-1, 14, 8, 8]               0
           Conv2d-19             [-1, 11, 6, 6]           1,397        --->to learn more patterns
      BatchNorm2d-20             [-1, 11, 6, 6]              22
             ReLU-21             [-1, 11, 6, 6]               0
           Conv2d-22             [-1, 10, 6, 6]             120        ---> to combine the features
        AvgPool2d-23             [-1, 10, 1, 1]               0
================================================================
Total params: 7,989
Trainable params: 7,989
Non-trainable params: 0

----------------------------------------------------------------
