# Explanation
The copy of s6 of s6.ipynb file holds the part i have done before S7 and later contains after S7 which is in final S6

## naive approach
copy of copy of s6.ipynb
conv1 - I have considered to have more kernels in the initial layer to to extact as many features as possible
            following s class i didnt add batchnorm initally since it might affect input features directly
conv2 - similar with the second layer to get more feature i convoloved again it with 3*3
pool1 - then to reduce channel and sum up the feature i used pool
conv3 - to increase receptive field
batch_norm - to normalize features , to learn effectively , to increase a contrast
conv4 - to combine similar channels
conv5 -to reduce the channel size
GAP - global avergae pooling to reduce it to only single channel size of 10 kernels

## after S7 approach
self.conv1 = nn.Conv2d(1,32,3)      --> to get as many features as possible
self.conv2 = nn.Conv2d(32,28,3)     --> to get as many features as possible
self.bt_nm1 = nn.BatchNorm2d(28)    --> to make next layers learn effectively
self.pool1 = nn.MaxPool2d(2,2)      --> to reduce channel size and approximate features
self.conv3 = nn.Conv2d(28,24,3)     --> to learn features
self.conv4 = nn.Conv2d(24,20,3)     --> to learn features
self.bt_nm = nn.BatchNorm2d(20)     --> to make next layers learn effectively
self.drop2 = nn.Dropout(0.15)       --> dropout to make every neuron learn all avriations
self.conv5 = nn.Conv2d(20,16,1)     --> combine channels
self.conv6 = nn.Conv2d(16,10,1)     --> reduce channels
self.gav1 = nn.AvgPool2d(8)         --> avg the channels


