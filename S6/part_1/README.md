#Part 1

![Screenshot (2)](https://github.com/ksharsha72/tsai/assets/90446031/f76c41d2-31f3-479f-ad31-a923075d865f)

The above screen shot is the excel file showing the differentiation with each and evey possible intermediate variable

The differentiation with every variable with other dependent variables are calculated and kept in differentaion_with_every_variable.xlsx

##The Probelm

![Screenshot (11)](https://github.com/ksharsha72/tsai/assets/90446031/bd572310-a4c8-4519-a1de-5f74ab45d463)

The Backpropogation in the neural network is the importatnt step where the networks learns the parameter,in order to learn it needs to know how the final computation is effected with small changes in all of its variables or values, this is where the differentation part comes, we push the final value to obtain certain value, and we make the network to change all the factored values to adjust them to get the desired final result

as shown in like above network we differentaite final loss with with every possiblw weight 

for example the final DE/DW5 looks how the changes to DE happens with a slightest change in W5 weight 

since W5 effected O1 and O1 affects a_O1, we apply chain rule of DE/DA_O1 * DA_01/D01 * DO1/DW5 , representing intermediate effects of the path, if we change the value of w5
