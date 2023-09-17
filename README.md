# PolyHunt
<H2> Hunt the Polynomial<br/></H2>
<H3>By: Apoorv Sharma</H3>
<H3>1. Method:<br/></H3>
<p>The Autofit method implemented in this assignment sweeps through order
until 30 and performs cross validation on the dataset for each order to get the average of
error from all folds. This error is stored in a list at the index, order -1.<br/>
The index which has the minimum error is returned, the order of the polynomial is
the returned index + 1.<br/>
For each order, the "regressing" function is called which divides the dataset into
training and test data, then it trains the model with training set and with the same model
the error for test set is calculated. The order which has the global minimum value of test error
is the order with best fit.<br/></p>

<H3>2. Regularization:<br/></H3>
<p>Yes, I have implemented regularization, which is basically the penalty value which
reduces the value of all weights to avoid over-fitting. The values which I tried were: 0.1, 0.001, 0.0005
for data where the value of 0.0005 give order for datasets A,B and C data set correctly.<br/></p>

<H3>3. Model Stability:<br/></H3>
<p>For me, i am getting the order values same for multiple runs, the weights are same as well</p>

<H3>4. Results:<br/></H3>
<H5>the following are the values of <U>known data sets(gamma = 0.0005)</U>:</H5>
A:<br/>
order of polynomial:  1<br/>
model weights:  [-1.17323472  1.56237056]<br/>
B:<br/>
order of polynomial:  2<br/>
model weights:  [-0.77192708  0.5576209  -1.52854281]<br/>
C:<br/>
order of polynomial:  4<br/>
model weights:  [-1.53110266  1.66272437  0.16329682  2.03459927 -0.80634571]<br/>
D:<br/>
order of polynomial:  6<br/>
model weights:  [-1.4227866   0.61412852 -1.09188766 -0.22067149  1.30339247 -0.4743951 -0.15902137]<br/>
E:<br/>
order of polynomial:  7<br/>
model weights:  [-0.3270503   0.93955684 -1.47017625 -1.00193291  0.98167268 -3.10695556 -2.42091091  4.61195976]<br/>
<p>The values we got for A,B and C match the order of secret provided, but for D and E, the values of order is
not equal. this might be because of more gaussian noise to these data which is not fitting with the regularization
from the other three datasets.</p>
<H5>the following are the values of <U>unknown data sets(gamma = 0.0005)</U>:</H5>
X:<br/>
order of polynomial:  5<br/>
model weights:  [ 0.32125533 -0.2076306   1.24099817  0.30960614 -0.22311238  1.21122788]<br/>
Y:<br/>
order of polynomial:  6<br/>
model weights:  [ 1.96101058  0.43767282 -0.0961212   0.26713046 -1.73262275 -0.33169969 4.25292221]<br/>
Z:<br/>
order of polynomial:  4<br/>
model weights:  [ 1.38592752  0.52801163  0.04664781 -0.80318654  0.08984255]<br/>
<br/>
The model and order values from these datasets might be correctbut it depends on the noise, providing correct 
regularization might give us correct output.<br/>

<H3>5. Collaboration:<br/></H3>
I have worked solo on this project.<br/>

<H3>6. Notes:<br/></H3>
Please refer to given below usage guidelines for running the file:<br/>
usage: ML_Project1_ApoorvSharma.py [-h] [-g] -tp  [-mo] [-I | --info | --no-info] [-nf] [-A | --autofit | --no-autofit] [-m]

Running the PolyHunt Project

options:<br />
  -h, --help:<t/>            show this help message and exit<br />
  -g , --gamma:          regularization constant (default as 0)<br />
  -tp , --trainPath:     filepath to the training data<br />
  -mo , --modelOutput:   a filepath where the best fit parameters will be saved<br />
  -I, --info, --no-info:
                        if this flag is set, name and contact information will be printed<br />
  -nf , --numFolds:      the number of folds to use for cross validation

autofitGroup:<br />
  -A, --autofit, --no-autofit:
                        flag which when supplied engages the order sweeping loop; if not supplied, fit the polynomial of the
                        given order<br />
  -m , --order:          polynomial order (or maximum order in autofit mode)<br />
<br/>
Some more pointers:<br/>
I have created an argument subgroup 'autofitGroup' which has two arguments autofit and order whose values are dependent 
on each other, if autofit is provided, that means we have to sweep though the degrees and find the best order. but if 
it is not provided, then order is mandatory.
