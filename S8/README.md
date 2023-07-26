# Code
The code contains the 3 model files, model,model_2,model_3 for  {Batcn_Norm,Layer_Norm,Group_Norm} correspondingly for CIFAR_10 DataSet

All the models were trained on the constraint of keeping the parameters < 50000

The Group Normalization and Layer Normalization  increased accuracy  a bit by 2-3%  compared to Batch Normalization, and it has been seen that the paremters were also low


The below images represent graphs for group_norm,batch_nom and layer_norm

![group_norm](https://github.com/ksharsha72/tsai/assets/90446031/b4670f65-6126-4e5a-8891-5dac5c0e3a80)

![batch_norm](https://github.com/ksharsha72/tsai/assets/90446031/51d94d0f-f917-4804-a79e-61d51f5a3967)

![layer_norm](https://github.com/ksharsha72/tsai/assets/90446031/b34bc47a-2138-4607-95c1-ee24e3f76ad5)


The accuracy achieved by the models are nearly 73-76%

here are some of the  mis-classified images and its prediction labels

![Screenshot (15)](https://github.com/ksharsha72/tsai/assets/90446031/788f3198-76ae-4dfb-a717-6f8783b2e8e7)![Screenshot (14)](https://github.com/ksharsha72/tsai/assets/90446031/c6d4ba1d-600d-4523-926c-a70a0cf5b81a)


![Screenshot (16)](https://github.com/ksharsha72/tsai/assets/90446031/6a1c111c-c68d-4d20-b431-26cfaf0de193)

utils.py contains show_kernels and plot_kernels, plot_kernels will check if no of channels >3 or not and if no of channels are >3 show_kernels will sum channels other wise it will print the kernels as its

