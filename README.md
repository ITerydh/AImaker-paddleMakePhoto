# 【AI创造营】Wechaty实用小工具---证件照助手

aistudio项目地址：
> [https://aistudio.baidu.com/aistudio/projectdetail/2253862](https://aistudio.baidu.com/aistudio/projectdetail/2253862)

<center><font size="5px" color="blue">你是否苦恼于没有条件将证件照换背景色</font></center>

<center><font size="4px" color="red">或是只有大头照。或是只有某种底色，要换其他底色，请使用它~</font></center>

<center><font size="4px" color="blue">那么请使用它~</font></center>

<center><font size="5px" color="blue">它就是你的证件照小助手</font></center>

<center><font size="3px" color="blue">（为了可玩性更高、所以不仅限于证件照）</font></center>

![示例](https://github.com/ITerydh/AImaker-paddleMakePhoto/blob/main/%E6%B5%81%E7%A8%8B.png)


**话不多说**

先搞个token再说   
>[http://pad-local.com](http://pad-local.com/#/)

## 1 配置服务器

我这里是搞了一台腾讯云的centos7.6的云服务器，其他也都一样的。    
在终端中依次粘贴入下列命令即可：    
（部分人可能自己搭建的有项目，8080端口存在被占用，随便换个808X端口就行了）

```
$ yum install docker

$ docker pull docker.io/wechaty/wechaty

$ export WECHATY_LOG="verbose"

$ export WECHATY_PUPPET="wechaty-puppet-wechat"

$ export WECHATY_PUPPET_SERVER_PORT="8080"

$ export WECHATY_TOKEN="puppet_padlocal_xxxxxx" # 这里输入你自己的token

$ docker run -ti --name wechaty_puppet_service_token_gateway --rm -e WECHATY_LOG -e WECHATY_PUPPET -e WECHATY_TOKEN -e WECHATY_PUPPET_SERVER_PORT -p "$WECHATY_PUPPET_SERVER_PORT:$WECHATY_PUPPET_SERVER_PORT" docker.io/wechaty/wechaty:latest
```

上述执行完成之后，在浏览器网址中输入你自己的地址：（换成自己的token）   
https://api.chatie.io/v0/hosties/puppet_padlocal_xxxxxxxx   
出现如下的字符即为成功！   
{"host":"xxx.xxx.xxx.xxx","ip":"xxx.xxx.xxx.xx","port":8080}   
![](https://ai-studio-static-online.cdn.bcebos.com/fbd1ba23df4a418293320594d103a3ee5dbe24c29b254af1900f16a11d90e4c6)   

## 2 登录二维码

在终端中出现如下的online链接，在浏览器中复制后打开，即可看到二维码，进行扫描登录即可！        
![](https://ai-studio-static-online.cdn.bcebos.com/abd2071b3e6040a4bbfbbc6bf2c1a0b858c8d8909844407082f7c9429cc64c44)



## 3 本地安装wechaty环境

aistudio无法开放端口，需要自行本地实现或者搭在服务器上

**项目仅为部分核心代码**


```python
# 0.8.15版本不太好使 切勿踩坑
!pip install wechaty==0.8.11
```

## 4 项目功能实现

项目主题为证件照生成，但为了功能不那么单一，也提供了其他api的功能

如：**情话、藏头诗、对联、天气查询**

### 4.1 证件照生成

采用paddlehub的人脸识别和人像分割得到图片

再进行换底！


```python
# 证件照生成函数
def makeZjz(img):
    pic_path = img
    face_landmark = hub.Module(name="face_landmark_localization")
    human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")

    rate = 1.3
    thresh = 2

    # 人脸识别
    result = face_landmark.keypoint_detection(paths=[pic_path])
    face = np.array(result[0]['data'][0], dtype=np.int64)
    # 剪裁
    left = face[:, 0].min()
    right = face[:, 0].max()
    w = right - left
    cw = int((right + left) / 2)
    upper = face[:, 1].min()
    lower = face[:, 1].max()
    h = lower - upper
    ch = int((lower + upper) / 2)
    h = int(413 * w / 295)
    box = (cw - rate * w, ch - rate * h, cw + rate * w, ch + rate * h)
    img = Image.open(pic_path)
    img = img.crop(box)
    img = img.resize((295, 413), Image.ANTIALIAS)


    result = human_seg.segmentation(images=[cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)],
                                    use_gpu=False,
                                    visualization=True,
                                    output_dir='humanseg_output')
    path = result[0]["save_path"]
    print("humanseg path:",path)

    # 上色
    for pic in ["red.png","blue.png","white.png"]:
        # 获取图片,方便后面的代码调用
        frame = Image.open(pic)
        # 给图片指定色彩显示格式
        logo = Image.open(path)
        logo = logo.convert("RGBA")
        frame.paste(logo, (0, 20), mask=logo)

        # 保存图片
        frame.save("result/final_"+pic)
        print("结果：result/final_"+pic)
```

### 4.2 情话 藏头诗 对联


```python
# Paddlehub文本模型
text_model1 = hub.Module(name='ernie_gen_lover_words')     # 情话模型
text_model2 = hub.Module(name="ernie_gen_acrostic_poetry", line=4, word=7)   # 藏头诗模型
text_model3 = hub.Module(name="ernie_gen_couplet")   # 获取对联下句

def chat_bot(content, mode):
    res = ''
    if mode == '0':
        res = get_content(content)
        # print(res)
    elif mode == '1':# 清话
        res = text_model1.generate(texts=[content], use_gpu=False, beam_width=1)
        if res is None:
            return
        res = res[0][0]
        # print(res)
    elif mode == '2':#藏头诗
        res = text_model2.generate(texts=[content], use_gpu=False, beam_width=1)
        if res is None:
            return
        res = res[0][0]
    elif mode == '3':#对联
        res = text_model3.generate(texts=[content], use_gpu=False, beam_width=1)
        out = []
        res = res[0][0]
    return res
```

### 4.3 地域天气


```python
# 获取城市天气
def get_weather_data(city_name):
    weatherJsonUrl = "http://wthrcdn.etouch.cn/weather_mini?city={}".format(city_name)  # 将链接定义为一个字符串
    response = requests.get(weatherJsonUrl)  # 获取并下载页面，其内容会保存在respons.text成员变量里面
    response.raise_for_status()  # 这句代码的意思如果请求失败的话就会抛出异常，请求正常就上面也不会做
    # 将json文件格式导入成python的格式
    weather_dict = json.loads(response.text)
    # print(weather_dict)
    if weather_dict['desc'] == 'invilad-citykey':
        weather_info = '请输入正确的城市名!'
    else:
        forecast = weather_dict.get('data').get('forecast')
        city = '城市：' + weather_dict.get('data').get('city') + '\n'
        date = '日期：' + forecast[0].get('date') + '\n'
        type = '天气：' + forecast[0].get('type') + '\n'
        wendu = '温度：' + weather_dict.get('data').get('wendu') + '℃ ' + '\n'
        high = '高温：' + forecast[0].get('high') + '\n'
        low = '低温：' + forecast[0].get('low') + '\n'
        ganmao = '感冒提示：' + weather_dict.get('data').get('ganmao') + '\n'
        fengxiang = '风向：' + forecast[0].get('fengxiang')
        weather_info = city + date + type + wendu + high + low + ganmao + fengxiang
    return weather_info
```

## 参考项目

不仅限于以下项目：

[【AI创造营+七夕特辑】七夕孤寡机器人](https://aistudio.baidu.com/aistudio/projectdetail/2263052)

[一步一步教你用wechaty+百度云主机打造一个带你穿越星际的微信机器人](https://aistudio.baidu.com/aistudio/projectdetail/2177502)

......

## 团队&&致谢

>团队：让我们水到底！    

> 本人： [iterhui](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/643467)    

>其他成员：[三岁](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/284366)、[super松](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/279448)、[L兮木](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/891283)、[七年期限](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/58637)（以上成员均是大佬，不分先后）

致谢：

感谢团队成员的各类建议和合作配合

感谢参考项目的各位大佬帮助

感谢积极帮助解决问题的各路大佬

最后感谢主办方的机会和精良的活动～～～

