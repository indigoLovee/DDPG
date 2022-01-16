# DDPG
 DDPG in Pytorch
 # 仿真环境
 gym中的LunarLanderContinuous-v2
 # 环境依赖
 * gym
 * numpy
 * matplotlib
 * python3.6
 * pytorch1.6
 # 文件描述
 * train.py为训练脚本，配置好环境后直接运行即可，不过需要在当前目录下创建output_images文件夹，用于放置生成的仿真结果；
 * network.py为网络脚本，包括演员网络和评论家网络；
 * buffer.py为经验回放池脚本；
 * DDPG.py为DDPG算法的实现脚本；
 * utils.py为工具箱脚本，里面主要放置一些通过函数；
 * test.py为测试脚本，通过加载训练好的权重在环境中进行测试，测试训练效果。
 # 仿真结果
 详见output_images文件夹
