#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pydantic import BaseModel


class BaseParameters(BaseModel):
    """基础参数，用户名；类别"""
    username: str
    category: str


class CheckStore(BaseParameters):
    """创建仓库请求的参数校验"""
    store_name: str


class UpdateTrainImages(CheckStore):
    """更新模型训练图片集请求的参数校验"""
    images_status: str
