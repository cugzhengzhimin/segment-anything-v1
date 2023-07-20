# Segment Anything "全家桶"
一个SAM的小实验，4月份熬了几天的夜，头发都少了几根，这里更加希望遇到为分割或者检测操碎心的小伙伴们。通用的一些数据集实验很多小伙伴都做了，因此这里主要想做一些其它数据集上的应用，以及优化SAM，原本很早就想做这件事情，但是被杂七杂八的破事给耽误了，前几天和北京的一个前同事吃饭，他一句话提醒了我，喜欢的事不一定要得到公司的支持，你可以自己支持自己，不然很多东西做了不就浪费了吗？后面我会以业余时间分享一些有趣的实验和结论，想法会在[知乎](https://www.zhihu.com/people/e-yu-jia-de-shuai-qi-zhu)上面更新，一些实验会陆续分享到GitHub，这里面没有涉密内容。

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`

## everything
```
python auto_predict.py
```

## everything 超大分辨率推理

```
python everything_infer.py
```


## 二阶段的推理
其中包括predict_fast_gpu_ssam_every和predict_fast_gpu_ssam两种提取方法实现单类别分割
```
python ssam_infer.py
```
