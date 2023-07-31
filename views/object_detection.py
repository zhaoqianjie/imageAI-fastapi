#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

from utils.model_operate import ModelOperate


async def start_train_model(data_directory, train_config):
    """
    开始训练模型
    :param data_directory: 数据集的路径
    :param train_config: 训练配置
    :return: None
    """

    async def train_model_():
        ModelOperate.train_model(data_directory, train_config)
        # time.sleep(60)

    await train_model_()


async def start_model_inference(inference_config):
    """
    开始训练模型
    :param inference_config: 推理配置
    :return: None
    """

    async def model_inference_():
        ModelOperate.model_inference(inference_config)
        # time.sleep(60)

    await model_inference_()
