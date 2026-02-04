#!/bin/bash

# 定义要运行的模型文件数组
models=("FBTCN预测模型.py" "OtherModel.py" "AnotherModel.py")

for model in "${models[@]}"
do
  for group in 1 2 3
  do
    for test_idx in 1 2 3 4 5
    do
      # 组名
      prefix="Bearing${group}_"
      # 测试集
      test="${prefix}${test_idx}"
      # context 只用 _1
      context="${prefix}1"
      # 训练集为该组除测试集外的所有
      train_list=""
      for i in 1 2 3 4 5
      do
        if [ "$i" != "$test_idx" ]; then
          train_list="${train_list}${prefix}${i},"
        fi
      done
      # 去掉最后一个逗号
      train_list=${train_list%,}

      echo "Running: python $model --train_xj=${train_list} --test_xj=${test} --context_xj=${context}"
      python $model --train_xj=${train_list} --test_xj=${test} --context_xj=${context}
    done
  done
done