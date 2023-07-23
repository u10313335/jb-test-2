# Problems

1. Use the linear model y = 2x +ε with zero-mean Gaussian noise ε∼ N(0, 1) to generate 15 data points with (equal spacing) x ∈ [−3, 3]

2. Perform **Linear Regression**. Show the fitting plot, the training error, and the five-fold cross-validation errors.

3. Perform **Polynomial Regression** with degree 5, 10 and 14, respectively. For each case, show the fitting plot, the training error, and the five-fold cross-validation errors. (Hint: Arrange the polynomial regression equation as follows and solve the model parameter vector w.)

   <img src="/home/kase/.config/Typora/typora-user-images/image-20230417010244835.png" alt="image-20230417010244835" style="zoom: 50%;" />

4. Change the model to y = sin(2πx) +ε with the noise ε∼ N(0, 0.04) and (equal spacing) x ∈ [0, 1]. Then repeat those stated in 2) and 3). Compare the results with linear/polynomial regression on different datasets.

5. Following 4), perform polynomial regression with degree 14 by varying the number of training data points m = 10, 80, 320. Show the five-fold cross-validation errors and the fitting plots. Compare the results to those in 4)

6. Following 4), perform polynomial regression of degree 14 via regularization. Compare the results by setting λ = 0, 0.001/m , 1/m, 1000/m , where m = 15 is the number of data points (with x = 0, 1/(m−1) , 2/(m−1), . . . , 1). Show the five-fold cross-validation errors and the fitting plots