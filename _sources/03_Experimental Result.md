#  2. Experimental Result
## 2.1 問題一

執行結果繪製成圖：

（藍點為產生的分佈點，黑線為 y = 2 * x）

![](https://hackmd.io/_uploads/Hy9tlRwq3.png)

## 2.2 問題二

### Fitting Plot

![](https://hackmd.io/_uploads/SJY9lRD5h.png)

* 藍色點：產生的 15 筆資料點
* 黑色線：y = 2 * x
* 藍色線：線性回歸的結果方程式

### Error:

![](https://hackmd.io/_uploads/HJBieADcn.png)

Training error:  0.051960923
5-fold cross validation error:  0.08670085109770298

## 2.3 問題三

### Fitting Plot

5 degree:
![](https://hackmd.io/_uploads/BJQneRvqn.png)

10 degree:
![](https://hackmd.io/_uploads/Bk33eRw9h.png)

14 degree:
![](https://hackmd.io/_uploads/HkHTlAPqn.png)

### Error:
![](https://hackmd.io/_uploads/BymAe0vcn.png)

**DEGREE 5:**

* Training error:  0.03820651
* 5-fold cross validation error:  7.849639892578125

**DEGREE 10:**

* Training error:  0.03717179
* 5-fold cross validation error:  49268.415625

**DEGREE 14:**

* Training error:  2.3874936
* 5-fold cross validation error:  2944648.8

## 2.4 問題四

### Fitting Plot:

**產生的資料點：**
![](https://hackmd.io/_uploads/S1byZCv93.png)

**Linear Regression （藍線）：**
![](https://hackmd.io/_uploads/S131bAvq3.png)

**5 Degree Ploynomial Regression（藍線）：**
![](https://hackmd.io/_uploads/rJPebCwch.png)

**10 Degree Ploynomial Regression（藍線）：**
![](https://hackmd.io/_uploads/SkWZ-Cw53.png)

**14 Degree Ploynomial Regression（藍線）：**
![](https://hackmd.io/_uploads/r1FZWRwcn.png)

### Error:
![](https://hackmd.io/_uploads/SyzzW0D9h.png)


*Linear Regression:*

* Training error:  0.24523492
* 5-fold cross validation error:   0.5493943139910697

*Polynmial Regression:*

* degree 5:
  * Training error:  8.9225614e-05
  * 5-fold cross validation error:  0.017055585980415344
* degree 10:
  * Training error:  8.30187
  * 5-fold cross validation error:  3.494831085205078
* degree 14:
  * Training error:  1.638919
  * 5-fold cross validation error:  23847.4140625

### On different set

多進行幾次實驗：
![](https://hackmd.io/_uploads/rk-mWRDc3.png)

![](https://hackmd.io/_uploads/By27-Rw5n.png)

![](https://hackmd.io/_uploads/HyY8-CPq3.png)

我們也可以看到，5 degree 的多項式表現得比線性回歸來得好。

## 2.5 問題五

### Fitting Plot:

#### 10 data
產生的資料點：
![](https://hackmd.io/_uploads/B1htb0v9h.png)

14 Degree Ploynomial Regression（藍線）：
![](https://hackmd.io/_uploads/BJ8qbAw92.png)

#### 80 data
產生的資料點：
![](https://hackmd.io/_uploads/ryXoZCD92.png)

14 Degree Ploynomial Regression（藍線）：
![](https://hackmd.io/_uploads/ryao-CPq2.png)

#### 320  data
產生的資料點：
![](https://hackmd.io/_uploads/HJOhWCD9h.png)

14 Degree Ploynomial Regression（藍線）：

![](https://hackmd.io/_uploads/B1fa-0wqn.png)

### Error:

![](https://hackmd.io/_uploads/H1yAZRw5h.png)

比較三組的 validation error：

* 10 data points: 4033.701171875
* 80 data points: 2946.8859375
* 320 data points: 8.50526123046875

## 2.6 問題六

### Fitting Plot:

產生的資料點：

![](https://hackmd.io/_uploads/S1n0WRv5n.png)

λ  = 0:

![](https://hackmd.io/_uploads/B1LJG0wq3.png)

λ  = 0.001/m:

![](https://hackmd.io/_uploads/SkggzAvqh.png)

λ  = 1/m:

![](https://hackmd.io/_uploads/SJ5gfRv5h.png)

λ  = 1000/m:

![](https://hackmd.io/_uploads/HyqWMAPqn.png)

### Error:

![](https://hackmd.io/_uploads/SJxff0D5n.png)

* **λ  = 0:** 24659.459375
* **λ  = 0.001/m:** 0.5945597171783448
* **λ  = 1/m:** 1.3468624114990235
* **λ  = 1000/m:** 0.8103070259094238