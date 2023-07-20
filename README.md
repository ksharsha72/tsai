<H1> Strcuture of the code </H1>
<ul>
<li>models.py</li>
<li>utils.py</li>
<li>s5.ipynb</li> 
</ul>
 

<h2>models</h2>
<p>the models.py hold the code that contains the model architecture</p>
<h2>utils</h2>
<p>the utils module contains all the import libraries which are common and also trian,test methods, which can also be confgiurable from notebook, transforms which are also common but thinking in a way ,it is better to move transforms to model.py because each model may have its own transforms</p>
<h2>s5.ipynb</h2>
<p>the s5 ipynb notebook contains all the imports of these modules and model execution in google colaab notebook</p>

<H1>How to run the code</H1>
clone the reposiotry,in to your folder and start executing the s5.ipynb notebook , which will import the models starts training and testing the accuracies
