# Part 1

![Screenshot (2)](https://github.com/ksharsha72/tsai/assets/90446031/f76c41d2-31f3-479f-ad31-a923075d865f)

The above screen shot is the excel file showing the differentiation with each and evey possible intermediate variable

The differentiation with every variable with other dependent variables are calculated and kept in differentaion_with_every_variable.xlsx

## The Probelm

![Screenshot (11)](https://github.com/ksharsha72/tsai/assets/90446031/bd572310-a4c8-4519-a1de-5f74ab45d463)

The Backpropogation in the neural network is the importatnt step where the networks learns the parameter,in order to learn it needs to know how the final computation is effected with small changes in all of its variables or values, this is where the differentation part comes, we push the final value to obtain certain value, and we make the network to change all the factored values to adjust them to get the desired final result

as shown in like above network we differentaite final loss with with every possiblw weight 

for example the final DE/DW5 looks how the changes to DE happens with a slightest change in W5 weight 

since W5 effected O1 and O1 affects a_O1, we apply chain rule of DE/DA_O1 * DA_01/D01 * DO1/DW5 , representing intermediate effects of the path, if we change the value of w5

**Differntiantion_with_every_variable.xlsx** hold backproogation , every variable with every other variable more than what is written

h1 = w1*i1 + w2*i2		
h2 = w3*i1 + w4*i2		
a_h1 = σ(h1) = 1/(1 + exp(-h1))		
a_h2 = σ(h2)		
o1 = w5*a_h1 + w6*a_h2		
o2 = w7*a_h1 + w8*a_h2		
a_o1 = σ(o1)		
a_o2 = σ(o2)		
E_total = E1 + E2		
E1 = ½ * (t1 - a_o1)²		
E2 = ½ * (t2 - a_o2)²		
![image](https://github.com/ksharsha72/tsai/assets/90446031/0e1504cf-245a-45f7-bc94-72da5576b274)


∂E_total/∂w5 = ∂(E1 + E2)/∂w5					
∂E_total/∂w5 = ∂E1/∂w5					
∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5					
∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)					
∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)					
∂o1/∂w5 = a_h1					
![image](https://github.com/ksharsha72/tsai/assets/90446031/0b43effc-b922-49b2-b51a-82bcf5768c0e)

∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1					
∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2					
∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1					
∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2					
![image](https://github.com/ksharsha72/tsai/assets/90446031/5d7897e5-c02a-4023-8f4f-6fc6d5a791a2)

∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1					
∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2					
∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3					
![image](https://github.com/ksharsha72/tsai/assets/90446031/cc51e330-73d0-4b4d-8b74-4da7dfe0a289)

∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1												
∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2												
∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1												
∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2												
![image](https://github.com/ksharsha72/tsai/assets/90446031/98dd82e2-ae9d-4887-8380-283f8ac09d84)






