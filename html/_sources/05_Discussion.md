# 4. Discussion

在實驗值中，大部分實驗結果均如假設，但有些實驗組別的表線超出我的想像，可能是出於我的學理知識不足，也可能是實驗瑕疵。

以下是我尚未清楚原因的現象：

* **training error > validation error?**

  ![image-20230419001522092](/home/kase/.config/Typora/typora-user-images/image-20230419001522092.png)

  做過幾次第四題的實驗，但是對於 10 degree polynomial 的 validation error 似乎都會小過 traininf data。
  
  
  
* **資料擴充效果，data = 10  好過 data = 80 ？**

  我們知道在機器學習中，資料是越多越好，這可以在資料筆數到達 320 時，表現明顯比其他兩組好得到證實。但是，在資料數 80 時，表現反而比 10 筆時還不理想：

  ![image-20230419005720898](/home/kase/.config/Typora/typora-user-images/image-20230419005720898.png)
  
  
  
* **正規化的 underfitting**

  課程簡報中的例子提到，當正規化使用的 λ 走到 1 ，模型會進入 overfitting。同時，隨著 λ 越來越大，這會是一個慢慢由 overfitting 轉往 underfitting 的過程，當經過那個可以產生最小誤差的 λ 後，誤差又會開始漸漸增大。但是這個現象在我的實驗中沒有看到，甚至是最後一組數據，當 λ 超越了 1，也沒有遇到 underfitting：

  ![image-20230419035020793](/home/kase/.config/Typora/typora-user-images/image-20230419035020793.png)



同時，我也覺得這次作業有許多不足之處：

* 程式架構不夠簡潔
* 圖畫得不好看（這點糾結許久，應該呈現局部的清晰細節，還是能看到函數整體樣貌，但粗糙的大圖。後來我選擇後者。）
* 對於實驗結果尚有疑慮