import asyncio
import base64
import io
import json
import os
import shutil
from zipfile import ZipFile
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from parameter_check import CheckStore, UpdateTrainImages
from status_code import Status
from views.object_detection import start_train_model, start_model_inference

app = FastAPI()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/creat-store")
async def creat_store(body: CheckStore):
    """
    创建仓库
    :param body: 请求体
    :return: JsonResponse
    """
    # todo：加入仓库个数限制
    target_path = os.path.join(DATA_PATH, body.username, body.category, body.store_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        return {"msg": Status.OK.msg, "code": Status.OK.code}
    else:
        return {"msg": Status.FILE_EXIST.msg, "code": Status.FILE_EXIST.code}


@app.get("/stores")
async def stores(username: str, category: str):
    """
    获取仓库列表
    :param username: 用户名
    :param category: 类别
    :return: JsonResponse
    """
    target_path = os.path.join(DATA_PATH, username, category)
    if not os.path.exists(target_path):
        return {"msg": Status.FILE_NOT_EXIST.msg, "code": Status.FILE_NOT_EXIST.code}

    store_list = os.listdir(target_path)
    return {"msg": Status.FILE_EXIST.msg, "code": Status.FILE_EXIST.code, "data": store_list}


def support_gbk(zip_file: ZipFile):
    """
    处理中文名乱码
    :param zip_file: zip文件对象
    :return: zip文件对象
    """
    name_to_info = zip_file.NameToInfo

    for name, info in name_to_info.copy().items():
        real_name = name.encode('cp437').decode('gbk')
        if real_name != name:
            info.filename = real_name
            del name_to_info[name]
            name_to_info[real_name] = info
    return zip_file


@app.post("/action/upload-zip")
async def upload_zip(file: UploadFile = File(),
                     username: str = Form(),
                     category: str = Form(),
                     store: str = Form(),
                     ):
    """
    上传zip文件
    :param file: zip文件
    :param username: 用户名
    :param category: 类别
    :param store: 仓库名
    :return: JsonResponse
    """
    # todo: 校验文件（大小校验，zip炸弹校验，内容校验）
    filename = file.filename
    ext = os.path.splitext(filename)[1]

    if not ext == ".zip":
        return {"msg": Status.FILE_FORMAT_ERROR.msg, "code": Status.FILE_FORMAT_ERROR.code}

    store_path = os.path.join(DATA_PATH, username, category, store)
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    # 解压到指定文件
    file_io = io.BytesIO(await file.read())
    with support_gbk(ZipFile(file_io)) as zf:
        images_list = zf.namelist()
        for file in images_list:
            zf.extract(file, store_path)

    # 数据库处理，新增图片记录
    # todo: 优化：批量新建，使用内存数据库
    # for image_name in images_list:

    return {"msg": Status.OK.msg, "code": Status.OK.code}


@app.get("/action/start-train")
async def start_train(username: str, category: str, store: str):
    """
    启动模型训练
    :param username: 用户名
    :param category: 类别
    :param store: 仓库名
    :return: JsonResponse
    """
    # 设置要训练网络的图像数据集的路径
    # todo: 参数校验；配置训练参数
    data_directory = os.path.join(DATA_PATH, username, category, store)
    classes_path = os.path.join(data_directory, "train", "annotations", "classes.txt")
    if not os.path.exists(classes_path):
        return {"msg": Status.FILE_NOT_EXIST.msg, "code": Status.FILE_NOT_EXIST.code}
    with open(classes_path, 'r') as f:
        classes = f.readlines()
    object_names_array = [i.strip() for i in classes]

    train_config = dict()
    train_config["data_directory"] = data_directory
    train_config["object_names_array"] = object_names_array
    train_config["batch_size"] = 4
    train_config["num_experiments"] = 10
    train_config["train_from_pretrained_model"] = "yolov3.pt"

    asyncio.create_task(start_train_model(data_directory, train_config))
    return {"msg": Status.OK.msg, "code": Status.OK.code}


@app.get("/action/start-inference")
async def start_inference(username: str, category: str, store: str):
    """
    启动模型推理
    :param username: 用户名
    :param category: 类别
    :param store: 仓库名
    :return: JsonResponse
    """
    # todo: 参数校验；配置推理参数
    store_path = os.path.join(DATA_PATH, username, category, store)
    inference_config = dict()
    inference_config['model_path'] = os.path.join(store_path, "models", "yolov3_jjjj_mAP-0.27718_epoch-16.pt")
    inference_config['configuration_json'] = os.path.join(store_path, "json", "jjjj_yolov3_detection_config.json")
    inference_config['input_dir'] = os.path.join(store_path, "inference")
    inference_config['output_dir'] = os.path.join(store_path, "detected", "images")

    asyncio.create_task(start_model_inference(inference_config))
    return {"msg": Status.OK.msg, "code": Status.OK.code}


@app.get("/detected-image")
async def detected_image(username: str, category: str, store: str, index: str):
    """
    获取推理图片
    :param username: 用户名
    :param category: 类别
    :param store: 仓库名
    :param index: 图片列表下标
    :return: JsonResponse
    """
    # todo:参数校验
    # 推理图片文件夹路径
    detected_images_dir = os.path.join(DATA_PATH, username, category, store, "detected")
    if not os.path.exists(detected_images_dir):
        return {"msg": Status.FILE_NOT_EXIST.msg, "code": Status.FILE_NOT_EXIST.code}

    images = os.listdir(detected_images_dir)
    image_index = int(index)
    if image_index <= 0:
        image_index = 0
    if image_index >= len(images) - 1:
        image_index = len(images) - 1

    image_path = os.path.join(detected_images_dir, images[image_index])
    image_name = os.path.basename(image_path)
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # todo: 优化：静态资源
        image_base64 = base64.b64encode(image_bytes)
        return {"msg": Status.OK.msg, "code": Status.OK.code,
                "data": {"image_base64": image_base64, "index": image_index, "total": len(images),
                         "image_name": image_name}}

    return {"msg": Status.FILE_NOT_EXIST.msg, "code": Status.FILE_NOT_EXIST.code}


@app.post("/action/update-train-images")
async def update_train_images(body: UpdateTrainImages):
    """
    更新模型训练的图片集
    :param body: 请求体
    :return: JsonResponse
    """
    # todo：加入仓库个数限制
    store_path = os.path.join(DATA_PATH, body.username, body.category, body.store_name)
    if not os.path.exists(store_path):
        return {"msg": Status.FILE_NOT_EXIST.msg, "code": Status.FILE_NOT_EXIST.code}

    images_status = json.dumps(body.images_status)
    pass_images = [image for image in images_status if images_status[image] == "Success"]
    # todo:issue: pass_images中如果和训练集的图片重名，会把训练集图片和标注覆盖掉
    for image in pass_images:
        annot = os.path.splitext(image)[0] + '.xml'
        src_image_path = os.path.join(store_path, 'detected', 'images', image)
        src_annot_path = os.path.join(store_path, 'detected', 'annotations', annot)

        dst_image_path = os.path.join(store_path, 'train', 'images', image)
        dst_annot_path = os.path.join(store_path, 'train', 'annotations', annot)

        if os.path.exists(src_image_path) and os.path.exists(src_annot_path):
            shutil.move(src_image_path, dst_image_path)
            shutil.move(src_annot_path, dst_annot_path)

    return {"msg": Status.OK.msg, "code": Status.OK.code}


if __name__ == '__main__':
    uvicorn.run(
        app='main:app',
        host="0.0.0.0",
        port=8899,
        workers=4,
        reload=True,
    )
