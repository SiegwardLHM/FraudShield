1. Backend 为后端文件夹 
2. Frontend 为前端文件夹

说明如下：
   
本团队研发的“一盾当关”系统，可以简单地划分为两个框架：前端和后端。

1. 前端：    

    前端可以在本地电脑运行，注意，需要安装django 、vue、python、npm 和 vscode。

    进入文件夹后，运行如下所示：

    npm run dev。

    还需注意，若有启动正常功能，还得创建网易云信账号，获取音视频通话的key，更新src/config/index.js文件，才能正常进行音视频通话。

    若要使用大模型助手，需要在后端文件里更换自己的openai的key。

2. 后端：

    后端需要部署在服务器上运行，建议安装conda环境，根据给出的environment.yml文件进行安装虚拟环境。后可直接运行

    检验：服务器GPU配置不能低于3090，cuda版本不低于12.0，否则将影响代码正常运行。

    运行代码：

    conda activate tzb

    python manage.py makemigrations

    python manage.py runserver