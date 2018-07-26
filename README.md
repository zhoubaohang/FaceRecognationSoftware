# 人脸识别软件(无外部API)

基于PCA模型设计的人脸识别软件

- 起初，在数据库课程设计部分，我用C#设计了一个基于人脸识别的签到记录系统。这个系统中的人脸识别部分使用的是商汤科技 Face++ API。由于识别速度比较慢，还需要连网，我总想将其改为离线式识别。所以，就有了这一篇博客。
- 基于之前写过的一篇博客 《图像处理之人脸识别》 中介绍的基于PCA的训练模型方法得到了人脸模型。
- 基于人脸模型、Python设计了这个小软件。这样摆脱了使用第三方API的缺点，识别速度更快。不过，经过测试，该方法受到光照影响较大。

## 开发工具

**环境**
1. windows 10
2. Anaconda(Spyder)

**语言**
1. 软件设计：Python
2. 模型训练使用：Matlab

**框架**
1. python-opencv
    - 实现人脸检测功能，得到人脸区域
2. numpy
    - 矩阵运算
3. scipy
    - 科学计算，加载模型文件
4. tkinter
    - GUI开发

## 功能简介

### 人脸识别

- 在本软件设计中，我们使用的模型文件为 Matlab 导出的 .mat 文件。文件里面保存了两个矩阵 **mean_face** 与 **V**，前者为 平均脸向量，后者为人脸空间矩阵。

- 该软件保存的用户人脸图像大小为 112 x 92。每次开启软件时，加载所有用户图像进入内存，并将二维图像拉伸为一维向量。
$v_{user}^{(i)}$代表用户$i$的人脸图像向量

- 然后，我们将所有用户图像向量组合为用户图像矩阵，该矩阵的每一列为用户图像向量：

$$U = \begin{bmatrix}
(v_{user}^{(0)})^T \ (v_{user}^{(1)})^T \ \cdots \ (v_{user}^{(n)})^T
\end{bmatrix}$$

- 将用户图像矩阵$U$中的每一列减去平均脸向量$v_{mean \_ face}$，再将运算后的矩阵投影至模型空间更新矩阵$U$:

$$U = V^T \cdot (U .- \ v_{mean \_ face})$$

- 如此一来，我们得到了降维后的用户人脸矩阵。

**识别过程**

1. 采集人脸图像，提取人脸部分，并将图像转换为向量形式：$v_{input}$

2. 将上一步得到的人脸向量按如下公式投影至模型空间：
$$v_{pca}=V^T \cdot (v_{input} - v_{mean \_ face})$$

3. 将上一步得到的$v_{pca}$向量与$U$矩阵中的每一列计算 欧式距离，找到最近的一列即为识别目标。

**识别函数代码**
``` python
def __recognize(self, image, face):
        """
        the system approves the user's identity according to his face
        """
        name = ''
        try:
            (x, y, w, h) = face
            image = Image.fromarray(image).crop((x, y, x+w, y+h)).resize(self.img_size)
            img_vec = self.V.T.dot(np.array(image).reshape([-1, 1]) - self.mean_face)
            distances = [la.norm(img_vec - self.user_matrix[:, j].reshape([-1, 1])) \
                         for j in range(self.user_matrix.shape[1])]
            
            min_dis = np.min(distances)
            index = np.where(distances == min_dis)[0][0]
            # print(min_dis, index)
            name = self.user_names[index]
        except:
            pass
        
        return name
```

### 人像导入

- 主要是为了方便导入用户人像，故加入该功能。

- 在界面中选择导入文件夹路径后，循环处理文件夹中的所有图像。提取人像部分并转换为灰度图片，保存至软件存储人像的相对路径下。

### 拍照录入

- 为了录入用户人像信息，用户可以在开启摄像头、输入姓名后，点击界面上的拍照按钮，即可保存人像信息至软件文件夹下。

## 软件截图

![Alt text](2018-07-26-人脸识别.png)

## 软件缺陷

1. 光照问题
    - 比如：在光线比较亮的地方录入人像后，用户在光线暗的地方就容易被识别错误。
    - 尝试解决问题的办法是：图像预处理部分使用了直方图均衡化，不过只起到了一部分作用。
    - 因此使用时，应尽可能地保持录入环境与检测环境一致。

