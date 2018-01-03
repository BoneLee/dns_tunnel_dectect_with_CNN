# dns_tunnel_dectect_with_CNN
dns tunnel dectect with CNN model

通过深度学习CNN模型来检测DNS隐蔽通道，见[《AI安全初探——利用深度学习检测DNS隐蔽通道》](http://www.freebuf.com/articles/network/158163.html)

---
使用方法：

(1) 安装requirements lib：pip install -r requirements.txt

(2) 训练模型: python dns_tunnel_train_model.py

(3) 预测xshell： python dns_tunnel_predict_xshell.py 
 
**注：dns tunnel工具的DNS流量样本收集请参考 [《DNS隐蔽通道检测——数据收集，利用iodine进行DNS隐蔽通道样本收集》](http://www.cnblogs.com/bonelee/p/8081744.html)。**

