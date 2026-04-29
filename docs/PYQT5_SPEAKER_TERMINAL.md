# PyQt5 声纹身份验证终端

## 1. Windows 侧环境

建议直接在现有项目 Python 环境里安装 GUI 依赖：

```bash
pip install -r requirements_gui.txt
```

如果当前环境还没有项目基础依赖，还需要保证至少能导入：

- `numpy`
- `torch`
- `torchaudio`
- `PyAV` 或其他可解码 `m4a/mp3` 的后备方案

## 2. 板子侧准备

先把下面两个文件拷到板子 `/root/models/`：

- `resnet_v7_best_fixed_rt160_nonorm.rknn`
- `rknn_embed_once.py`

例如 U 盘挂载点是 `/mnt/udisk`：

```bash
cp /mnt/udisk/resnet_v7_best_fixed_rt160_nonorm.rknn /root/models/
cp /mnt/udisk/rknn_embed_once.py /root/models/
```

## 3. 网络配置

PC 与板子网线直连后：

- Windows：`192.168.50.1/24`
- Board：`192.168.50.2/24`

板子临时设置：

```bash
ip link set eth0 up
ip addr add 192.168.50.2/24 dev eth0
```

Windows 验证：

```powershell
ping 192.168.50.2
```

## 4. 启动 GUI

```bash
python speaker_identity_terminal.py
```

## 5. 默认板端配置

- Host: `192.168.50.2`
- User: `root`
- Port: `22`
- Model: `/root/models/resnet_v7_best_fixed_rt160_nonorm.rknn`
- Embed script: `/root/models/rknn_embed_once.py`
- Threshold: `0.187723`
