# Explanation
The copy of s6 of s6.ipynb file holds the part i have done before S7 and later contains after S7 which is in final S6

## naive approach
copy of copy of s6.ipynb <br/>
conv1 - I have considered to have more kernels in the initial layer to to extact as many features as possible <br/>
            following s class i didnt add batchnorm initally since it might affect input features directly <br/>
conv2 - similar with the second layer to get more feature i convoloved again it with 3*3 <br/>
pool1 - then to reduce channel and sum up the feature i used pool <br/>
conv3 - to increase receptive field <br/>
batch_norm - to normalize features , to learn effectively , to increase a contrast <br/>
conv4 - to combine similar channels <br/>
conv5 -to reduce the channel size <br/>
GAP - global avergae pooling to reduce it to only single channel size of 10 kernels <br/>

## after S7 approach
Final S6 with 19k params able to achieve 99.4 to 5 in last 6-7 epochs <br/>
self.conv1 = nn.Conv2d(1,32,3)      --> to get as many features as possible <br/>
self.conv2 = nn.Conv2d(32,28,3)     --> to get as many features as possible <br/>
self.bt_nm1 = nn.BatchNorm2d(28)    --> to make next layers learn effectively <br/>
self.pool1 = nn.MaxPool2d(2,2)      --> to reduce channel size and approximate features <br/>
self.conv3 = nn.Conv2d(28,24,3)     --> to learn features <br/>
self.conv4 = nn.Conv2d(24,20,3)     --> to learn features <br/>
self.bt_nm = nn.BatchNorm2d(20)     --> to make next layers learn effectively <br/>
self.drop2 = nn.Dropout(0.15)       --> dropout to make every neuron learn all avriations <br/>
self.conv5 = nn.Conv2d(20,16,1)     --> combine channels <br/>
self.conv6 = nn.Conv2d(16,10,1)     --> reduce channels <br/>
self.gav1 = nn.AvgPool2d(8)         --> avg the channels <br/>


