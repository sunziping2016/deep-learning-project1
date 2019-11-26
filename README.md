# 如何运行

先安装依赖。

```shell script
pip install -r requirements.txt
```

以下命令训练。其中`<gpu_ids>`是逗号分隔的GPU id列表，如`0,2,3`，如果不指定会在CPU上运行。`<name>`是这次训练的模型的名字，强烈建议指定，默认叫`test`。


```shell script
python main.py --epoch 1000 --gpu <gpu_ids> --save_path "save/<name>"
```

训练的时候，你可以打开TensorBoard查看loss和准确率。

```shell script
tensorboard --logdir runs
```

以下命令在validation数据集上测试，`<epoch_id>`是第几个epoch，应该从1开始。

```shell script
python main.py --load <epoch_id> --task valid
```

以下命令生成用于提交的`save/<name>/predictions.txt`，它是test数据集上的结果`。

```shell script
python main.py --load <epoch_id> --task test --save_path "save/<name>"
```

# 如何改进

如果需要调整batch size、learning rate之类的参数，可以直接通过命令行指定。如果需要调整模型的参数，如隐藏层元素个数，建议将`model/<model_name>.py`的需要调整的参数写到构造函数参数里，再用`parser.add_argument`创建新的命令行参数，然后传递给模型构造函数。如果需要添加新架构的模型，请在`model/`下新建文件，然后仿照`ImageNetCNN`的写法添加到`main.py`中。
