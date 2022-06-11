Development of solubility measurement technology using computer vision
===
Software Convergence Capstone design 2022-1
---

- **Minwoo Jeon (2017103752, software convergence Kyunghee Univs.)**
- Advisor Prof. Hyoseok Hwang(software convergence Kyunghee Univs.)


- - -

## Overview
* oled 및 반도체 제작 공정 중 플라스크 안의 유기용매의 용질이 녹은 정도인 용해도를 측정하는 과정이 있다. 현재 해당 과정은 사람이 직접 눈으로 일일이 확인함에 따라 완전한 자동화가 이루어지지않고 있다. 용해도 측정을 컴퓨터 비전 및 딥러닝을 이용하여 자동으로 수행하는 것은 end-to-end automation 제작 공정 달성에 있어 필수적인 요소이다.

* Evaporating flask에 담긴 용액(100ml)의 용해도를 컴퓨터 비전을 이용하여 측정한다. 실제 반도체 제작공정에서 쓰일 용질(CuOAc, PdOaC, CuSO, CuBr)과 증류수(100ml)으로 구성된 용액을 준비하고, 다양한 background image(Grid bg, white bg) 환경에서 데이터 셋을 구현한다. 

* 얻어진 데이터 셋을 이용하여 image processing(Circular ROI extraction, Grid homogeneity analysis, Radial profiling, Particle segmentation, Check pattern detection) 과정을 거쳐 9가지 feature(mmg, msg, smg, ssg, std range, curve_c, gradient, particle sum, superposition ratio)을 얻는다. 9가지 feature을 MLP 및 linear SVM model을 이용하여 train시켜 classifier를 구현한다.   

* System outline   
<img width="633" alt="image" src="https://user-images.githubusercontent.com/65657711/173188232-dfd7b6df-3924-4033-94db-51aff5ce14a4.png">
 
* Block diagram    
![image](https://user-images.githubusercontent.com/65657711/173181388-28d475db-4e68-4019-a4fa-2096bbf0be23.png)

* Analysis algorithm   
![image](https://user-images.githubusercontent.com/65657711/173181403-7a90ad65-271e-4e3c-af44-502700d08da6.png)
![image](https://user-images.githubusercontent.com/65657711/173181404-8ee4e27f-71d2-41fa-9d81-f4b03aad27fd.png)

- - -
## Step-by-step outputs
#### 1) Circular ROI extraction   
![image](https://user-images.githubusercontent.com/65657711/173181542-731b436d-a162-4d26-b308-0fae89ee3c8e.png)
![image](https://user-images.githubusercontent.com/65657711/173181541-10736089-dab4-4f00-878b-ae8944d7c68c.png)

#### 2) Grid homogeneity analysis
![image](https://user-images.githubusercontent.com/65657711/173181603-5c74f976-9047-42a7-a590-0b35574419b8.png)

[PdOAc(0.1g/ml)]   
mean of mean of grid(**mmg**): 164.410398   
mean of std of grid(**msg**): 26.402044   
std of mean of grid(**smg**): 28.279761   
std of std of grid(**ssg**): 8.267754

#### 3) Radial profiling
<img width="1015" alt="image" src="https://user-images.githubusercontent.com/65657711/173181739-47c586c4-e7b7-42c1-8d1d-f23ab5575f20.png">

[PdOAc(0.1g/ml)]   
curve coefficient: a- 9.11485524e-04, b- 4.53656349e-01, **c- 1.92287436e+02**   
최소, 최대값 좌표 & **기울기**: (0 192), (249 123), **0.2771**   
**std range: 24465.72945018559**   

#### 4) Particle Segmentation
![image](https://user-images.githubusercontent.com/65657711/173181779-fac2c857-f77f-4762-aa90-b9bf2e87a98a.png)
![image](https://user-images.githubusercontent.com/65657711/173181770-37cfa140-b1a3-49a1-b1f3-a11de5356562.png)
![image](https://user-images.githubusercontent.com/65657711/173181775-61e2955e-fa6c-4e9c-8009-cafb6383296e.png)

[PdOAc(0.1g/ml)]
**Particle sum: 39182**

#### 5) Check pattern detection
![image](https://user-images.githubusercontent.com/65657711/173181802-ecffc613-b7dc-49d5-94bc-528805c36f58.png)

![image](https://user-images.githubusercontent.com/65657711/173181808-334a54d8-75e5-46a3-bad6-87746ceae726.png)
![image](https://user-images.githubusercontent.com/65657711/173181813-c45d5749-35f7-445e-b766-43a69f098766.png)
![image](https://user-images.githubusercontent.com/65657711/173181817-04c18c58-0e72-4e9b-a44d-a3d66a269426.png)

[CuOAc(0.5g/100ml)]   
sum of grid mask: 2429640   
sum of superposition: 1855890   
**supp ratio: 76.39**

#### 6) MLP & linear SVM 

#### [MLP (ReLU Activateion function, SGD optimizer, Cross Entropy Loss fuction)]
<img width="700" alt="image" src="https://user-images.githubusercontent.com/65657711/173181948-936dd9f7-a57d-4239-b6ce-95a74beb3591.png">

**[MLP 4-Fold validation output]**   
Fold 0: 94.11764705882352 %   
Fold 1: 88.23529411764706 %   
Fold 2: 100.0 %   
Fold 3: 94.11764705882352 %   
**Average: 94.11764705882354 %**

#### linear SVM (gamma = 0.1, C=0.05)
**[SVM 4-Fold validation output]**
Fold 0: 100.0 %   
Fold 1: 94.11764706 %   
Fold 2: 100.0 %   
Fold 3: 94.11764706 %   
**Average: 97.05882352941177 %**


- - -
## Results
* Calibration   
<img width="505" alt="image" src="https://user-images.githubusercontent.com/65657711/173186043-c26b6ae5-cb1f-42c2-bc3d-6bdc49edcfab.png">

* White BG image Capture & Processing   
<img width="505" alt="image" src="https://user-images.githubusercontent.com/65657711/173186056-b67d2087-6512-4c75-8f9a-9cd4ec470778.png">

* Grid BG image Cature & Processing & Classify   
<img width="505" alt="image" src="https://user-images.githubusercontent.com/65657711/173186074-118fa284-eff0-4554-8808-9f9d168a9c4f.png">









