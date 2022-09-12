如题，部分参考  [深度学习——Ubuntu20.04服务器版本系统安装，及NVIDIA显卡驱动和PyTorch环境完整安装教程](https://blog.csdn.net/weixin_41945051/article/details/108481879?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.1&utm_relevant_index=3 "深度学习——Ubuntu20.04服务器版本系统安装，及NVIDIA显卡驱动和PyTorch环境完整安装教程")

注意：安装的Ubuntu版本若是自带gui，会和lightdm冲突，进入命令行界面修复一下就可以。开机时按下esc键进入ububtu高级选项，选择第一个选项即可进入命令行模式。

```shell
sudo dpkg-reconfigure gdm3
```