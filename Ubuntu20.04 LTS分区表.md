### **双硬盘**
#### **512g 固态 + 2t机械**

把512固态全部作为启动盘，设置如下：
efi文件系统分区 4g (主分区)
swap交换分区 32g (逻辑分区)
/根目录 ext4 余下空间(逻辑分区)
留100g空间备用

2t机械全部设置成ext4 /home挂载(逻辑分区)

注意：单系统引导直接装到efi分区，双系统则新增/boot分区，引导分区都需要设置成主分区！！

