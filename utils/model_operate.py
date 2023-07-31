#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict

from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection


class ModelOperate(object):

    @staticmethod
    def train_model(data_directory: str, train_config: Dict):
        """
        模型训练
        :param data_directory:
        :param train_config:
        :return: None
        """
        # 定义模型训练器
        trainer = DetectionModelTrainer()
        # 设置网络类型
        trainer.setModelTypeAsYOLOv3()
        # 设置要训练网络的图像数据集的路径
        trainer.setDataDirectory(data_directory=data_directory)
        """
        num_objects ：这是一个数组，其中包含我们数据集中对象的名称
        batch_size：这是为了说明训练的批量大小。
        默认batch_size为 4。如果您正在使用Google Colab进行训练，那会很好。
        但是，我建议您使用比 Colab 提供的 K80 更强大的 GPU，因为您的batch_size越高 （8， 16），您的检测模型的准确性就越好。
        num_experiments：这是为了说明网络将训练所有训练图像的次数， 这也被称为epochs
        train_from_pretrained_model（可选）：这是使用来自预训练的 YOLOv3 模型的迁移学习进行训练
        """
        trainer.setTrainConfig(object_names_array=train_config["object_names_array"],
                               batch_size=train_config["batch_size"],
                               # num_experiments=200,
                               num_experiments=train_config["num_experiments"],
                               train_from_pretrained_model=train_config["train_from_pretrained_model"])

        # 模型将保存在 ’data_directory‘/models文件夹中。mAP50 越高，模型越好
        trainer.trainModel()

    @staticmethod
    def model_inference(inference_config: Dict):
        """
        启动推理
        :param inference_config: 推理配置
        :return: None
        """

        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()  # 指定推理模型
        detector.setModelPath(model_path=inference_config['model_path'])  # 模型文件路径
        # configuration_json （必需）：这是 .json 文件的路径
        detector.setJsonPath(configuration_json=inference_config['configuration_json'])
        detector.loadModel()  # 加载模型
        input_dir = inference_config['input_dir']
        output_dir = inference_config['output_dir']
        inference_images_list = os.listdir(input_dir)
        inference_images_path = [os.path.join(input_dir, i) for i in inference_images_list]
        for image_path in inference_images_path:
            image_name = os.path.basename(image_path)
            primary, ext = os.path.splitext(image_name)
            output_image_path = os.path.join(output_dir, primary + "_detected" + ext)
            detections = detector.detectObjectsFromImage(input_image=image_path, output_image_path=output_image_path)

            # todo: 生成xml标注文件
            for detection in detections:
                print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
