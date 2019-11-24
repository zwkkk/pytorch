# 优化器

参考：[https://blog.csdn.net/shanglianlm/article/details/85143614](https://blog.csdn.net/shanglianlm/article/details/85143614)

PyTorch学习率调整策略通过torch.optim.lr_scheduler接口实现。PyTorch提供的学习率调整策略分为三大类，分别是

- 有序调整：等间隔调整(Step)，按需调整学习率(MultiStep)，指数衰减调整(Exponential)和 余弦退火CosineAnnealing。

- 自适应调整：自适应调整学习率 ReduceLROnPlateau。

- 自定义调整：自定义调整学习率 LambdaLR。

  

1. 等间隔调整学习率 StepLR

   等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了。

```python
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

'''
参数：

step_size(int) 
学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。

gamma(float)
学习率调整倍数，默认为 0.1 倍，即下降 10 倍。

last_epoch(int)
上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值。
'''
```



2. 按需调整学习率 MultiStepLR

按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。

```python
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

'''
参数：

milestones(list)
一个 list，每一个元素代表何时调整学习率， list 元素必须是递增的。如 milestones=[30,80,120]

gamma(float)
学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
'''
```



3. 指数衰减调整学习率 ExponentialLR

按指数衰减调整学习率，调整公式: lr=l∗gamma∗∗epoch

```python
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
'''
参数：

gamma
学习率调整倍数的底，指数为 epoch，即 gamma**epoch
'''
```



4. 余弦退火调整学习率 CosineAnnealingLR

以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2∗Tmax
2∗Tmax 为周期，在一个周期内先下降，后上升。

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
'''
参数：

T_max(int)
一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。

eta_min(float)
最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。
'''
```



5. 自适应调整学习率 ReduceLROnPlateau

当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。
例如，当验证集的 loss 不再下降时，进行学习率调整；或者监测验证集的 accuracy，当accuracy 不再上升时，则调整学习率。

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


'''
参数：

mode(str)
模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。

factor(float)
学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor

patience(int)
忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。

verbose(bool)
是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))

threshold_mode(str)
选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold；

threshold(float)
配合 threshold_mode 使用。

cooldown(int)
“冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。

min_lr(float or list)
学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。

eps(float)
学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
```



6. 自定义调整学习率 LambdaLR

为不同参数组设定不同学习率调整策略。调整规则为，

lr=base_lr∗lmbda(self.last_epoch)
lr=base_lr∗lmbda(self.last_epoch)

fine-tune 中十分有用，我们不仅可为不同的层设定不同的学习率，还可以为其设定不同的学习率调整策略。

```python
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

'''
参数：

lr_lambda(function or list)
一个计算学习率调整倍数的函数，输入通常为 step，当有多个参数组时，设为 list。
'''
```



## 使用方法

```python
#定义学习率策略
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)

for i in range(args.EPOCHS):
    cnn.train()
    x_train, y_train, x_test, y_test = data.next_batch(args.BATCH)  # 读取数据
    
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.long().to(device)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_test = x_test.float().to(device)
    y_test = y_test.long().to(device)

    outputs = cnn(x_train)
    _, prediction = torch.max(outputs.data, 1)

    optimizer.zero_grad()  
    loss = loss_fn(outputs, y_train)
    loss.backward()
    
    #############################################################
    
    scheduler.step(loss)  #参数进行更新,因此scheduler加在这里,记得加上评价指标loss
    
    #############################################################
    
    print(loss.detach())
    # 若测试准确率高于当前最高准确率，则保存模型
    train_accuracy = eval(model, x_test, y_test)
    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (i, best_accuracy))
```



