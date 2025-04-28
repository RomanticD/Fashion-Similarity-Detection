# src/app/image_upload_route.py
import logging
import time
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin, CORS
import json
from io import BytesIO

from src.app.supabse_route import admin_required
from src.core.groundingdino_handler import ClothingDetector
from src.db.uploads.image_upload import ImageUploader
from src.core.vector_index import VectorIndex
from src.utils.data_conversion import base64_to_numpy
from src.utils.request_tracker import request_tracker, CancellationException
from src.core.image_similarity import ImageSimilarity
from src.repo.split_images_repo import save_to_db

# 定义一个 Blueprint 来组织路由
api_up = Blueprint('image_upload', __name__)

CORS(api_up)

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取项目根目录
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

# 初始化所需组件 - 调整检测器的阈值使其更敏感
clothing_detector = ClothingDetector()
# 手动降低检测阈值，使其更容易识别衣物
clothing_detector.box_threshold = 0.15
image_uploader = ImageUploader()
vector_index = VectorIndex()


def ensure_rgb_format(image_np):
    """确保图像是RGB格式（3通道）"""
    # 检查图像的通道数
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        # 如果是RGBA (4通道)，转换为RGB (3通道)
        logger.info("将RGBA图像转换为RGB格式")
        img = Image.fromarray(image_np)
        rgb_img = img.convert('RGB')
        return np.array(rgb_img)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # 已经是RGB格式
        return image_np
    elif len(image_np.shape) == 2:
        # 如果是灰度图像（2维），转换为RGB
        logger.info("将灰度图像转换为RGB格式")
        img = Image.fromarray(image_np)
        rgb_img = img.convert('RGB')
        return np.array(rgb_img)
    else:
        # 处理其他不常见的格式
        logger.warning(f"不支持的图像格式: shape={image_np.shape}")
        # 尝试直接转换
        img = Image.fromarray(image_np)
        rgb_img = img.convert('RGB')
        return np.array(rgb_img)


@api_up.route("/upload_image", methods=["POST"])
@admin_required
@cross_origin()
def upload_image():
    """
    前端上传单张图片的接口

    请求格式:
    {
        "image_base64": "base64编码的图片字符串",
        "image_name": "可选的图片名称",
        "force_process": true/false (可选，是否强制处理即使没有检测到衣物)
    }

    返回格式:
    {
        "success": true/false,
        "message": "状态信息",
        "request_id": "请求ID，可用于取消请求",
        "data": {
            "original_image_id": "原图ID",
            "segments": [
                {
                    "splitted_image_id": "分割图片ID",
                    "splitted_image_path": "分割图片路径"
                },
                ...
            ]
        }
    }
    """
    start_time = time.time()

    # 生成唯一的请求ID
    request_id = str(uuid.uuid4())

    # 在追踪器中注册请求
    request_tracker.register_request(request_id)

    try:
        # 获取请求数据
        data = request.get_json()
        base64_image = data.get('image_base64')
        force_process = data.get('force_process', True)  # 默认强制处理

        # 如果未提供图片名称，则生成一个基于时间戳的唯一名称
        image_name = data.get('image_name', f"uploaded_{int(time.time())}")

        # 参数验证
        if not base64_image:
            logger.error("请求数据中缺少 'image_base64' 字段")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "缺少必要参数: 'image_base64'"
            }), 400

        logger.info(f"开始处理图片: {image_name}")

        # 步骤1: 将base64转换为图像数组
        try:
            convert_start = time.time()

            # 定义可取消的转换函数
            def convert_image():
                # 移除可能存在的数据前缀（例如 "data:image/jpeg;base64,"）
                if base64_image.startswith('data:'):
                    clean_base64 = base64_image.split(',', 1)[1]
                else:
                    clean_base64 = base64_image

                # 转换为numpy数组
                image_np = base64_to_numpy(clean_base64)

                # 确保图像是RGB格式（3通道）
                image_np = ensure_rgb_format(image_np)

                logger.info(f"图像转换用时: {time.time() - convert_start:.4f} 秒")
                logger.info(f"图像形状: {image_np.shape}")
                return image_np

            # 执行可取消操作
            image_np = request_tracker.run_cancellable(request_id, convert_image)

        except CancellationException:
            logger.info(f"请求 {request_id} 在图像转换过程中被取消")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "处理已被用户取消",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"图像转换错误: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"图像转换错误: {str(e)}",
                "request_id": request_id
            }), 400

        # 步骤2: 使用GroundingDINO检测服装物品
        try:
            detect_start = time.time()

            # 定义可取消的检测函数
            def detect_clothes():
                return clothing_detector.detect_clothes(image_np)

            # 执行可取消操作
            segmented_images = request_tracker.run_cancellable(request_id, detect_clothes)
            logger.info(f"服装检测用时: {time.time() - detect_start:.4f} 秒")

            # 检查是否检测到服装
            if not segmented_images and not force_process:
                request_tracker.complete_request(request_id)
                return jsonify({
                    "success": False,
                    "message": "未在图像中检测到服装物品",
                    "request_id": request_id
                }), 200

            # 如果没有检测到服装但强制处理，我们使用整个图像
            if not segmented_images and force_process:
                logger.warning("未检测到服装，但因强制处理标志而继续处理整张图像")
                # 将整张图像作为分割使用 - 确保是RGB格式
                segmented_images = [image_np.copy()]

        except CancellationException:
            logger.info(f"请求 {request_id} 在服装检测过程中被取消")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "处理已被用户取消",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"服装检测错误: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"服装检测错误: {str(e)}",
                "request_id": request_id
            }), 500

        # 步骤3: 创建数据目录存储分割图像
        data_dir = root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 为原图创建子目录
        image_dir = data_dir / image_name
        image_dir.mkdir(parents=True, exist_ok=True)

        # 步骤4: 处理并上传每个分割图像
        uploaded_segments = []
        try:
            upload_start = time.time()

            for idx, img_array in enumerate(segmented_images):
                # 检查请求是否被取消
                if request_tracker.is_cancelled(request_id):
                    logger.info(f"请求 {request_id} 在图像上传过程中被取消")
                    request_tracker.complete_request(request_id)
                    return jsonify({
                        "success": False,
                        "message": "处理已被用户取消",
                        "request_id": request_id
                    }), 200

                # 再次确保每个分割图像是RGB格式
                img_array = ensure_rgb_format(img_array)

                # 输出当前处理的图像形状，用于调试
                logger.info(f"处理分割图像 {idx}，形状: {img_array.shape}")

                filename = f"segment_{idx}.png"
                save_path = image_dir / filename

                # 处理并上传图像
                splitted_image_id = f"{image_name}_segment_{idx}"
                relative_path = f"{image_name}/{filename}"

                # 定义可取消的上传操作
                def upload_segment():
                    return image_uploader.process_and_upload_image(img_array, idx, save_path, image_name)

                # 执行可取消操作
                processed_path = request_tracker.run_cancellable(request_id, upload_segment)

                if processed_path:
                    uploaded_segments.append({
                        "splitted_image_id": splitted_image_id,
                        "splitted_image_path": relative_path
                    })

            logger.info(f"图像上传用时: {time.time() - upload_start:.4f} 秒")

        except CancellationException:
            logger.info(f"请求 {request_id} 在分割图像上传过程中被取消")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "处理已被用户取消",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"分割图像上传错误: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"分割图像上传错误: {str(e)}",
                "request_id": request_id
            }), 500

        # 步骤5: 更新向量索引
        # 注意: 在高并发环境下，每次上传后重建索引可能效率较低
        # 未来可以考虑实现增量更新或定期重建索引的机制
        try:
            index_start = time.time()

            # 定义可取消的索引更新操作
            def update_index():
                return vector_index.rebuild_index()

            # 执行可取消操作
            index_result = request_tracker.run_cancellable(request_id, update_index)

            if index_result[0] is None:
                logger.warning("向量索引更新失败，但图像已成功上传")

            logger.info(f"索引更新用时: {time.time() - index_start:.4f} 秒")

        except CancellationException:
            logger.info(f"请求 {request_id} 在索引更新过程中被取消")
            # 注意: 即使索引更新被取消，我们仍然认为上传成功
            # 因为图像和向量数据已经保存到数据库中
        except Exception as e:
            logger.error(f"索引更新错误: {e}")
            # 仍然返回成功，只是带有警告信息

        # 标记请求为完成
        request_tracker.complete_request(request_id)

        total_time = time.time() - start_time
        logger.info(f"总处理时间: {total_time:.4f} 秒")

        return jsonify({
            "success": True,
            "message": f"成功处理并上传图像，检测到 {len(uploaded_segments)} 个服装分割",
            "request_id": request_id,
            "data": {
                "original_image_id": image_name,
                "segments": uploaded_segments
            }
        })

    except Exception as e:
        logger.error(f"意外错误: {e}")
        request_tracker.complete_request(request_id)
        return jsonify({
            "success": False,
            "message": f"意外错误: {str(e)}",
            "request_id": request_id
        }), 500


# 添加取消正在进行的上传的接口
@api_up.route("/cancel_upload/<request_id>", methods=["POST"])
@cross_origin()
def cancel_upload(request_id):
    """
    取消正在进行的图像上传请求
    """
    success = request_tracker.cancel_request(request_id)
    if success:
        return jsonify({
            "success": True,
            "message": f"请求 {request_id} 已标记为取消"
        }), 200
    else:
        return jsonify({
            "success": False,
            "message": f"请求 {request_id} 未找到或已完成"
        }), 404


@api_up.route("/image/split", methods=["POST"])
@cross_origin()
def split_image():
    """
    将长图分割成多个服装图片的接口

    请求格式:
    {
        "image_base64": "base64编码的图片字符串"
    }

    返回格式:
    {
        "success": true/false,
        "message": "状态信息",
        "segments": [
            {
                "image_base64": "分割后的base64图片字符串"
            },
            ...
        ]
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        base64_image = data.get('image_base64')

        # 参数验证
        if not base64_image:
            logger.error("请求数据中缺少 'image_base64' 字段")
            return jsonify({
                "success": False,
                "message": "缺少必要参数: 'image_base64'"
            }), 400

        # 步骤1: 将base64转换为图像数组
        try:
            # 移除可能存在的数据前缀
            if base64_image.startswith('data:'):
                clean_base64 = base64_image.split(',', 1)[1]
            else:
                clean_base64 = base64_image

            # 转换为numpy数组
            image_np = base64_to_numpy(clean_base64)

            # 确保图像是RGB格式
            image_np = ensure_rgb_format(image_np)

        except Exception as e:
            logger.error(f"图像转换错误: {e}")
            return jsonify({
                "success": False,
                "message": f"图像转换错误: {str(e)}"
            }), 400

        # 步骤2: 使用GroundingDINO检测服装物品
        try:
            segmented_images = clothing_detector.detect_clothes(image_np)

            if not segmented_images:
                return jsonify({
                    "success": False,
                    "message": "未在图像中检测到服装物品"
                }), 200

            # 将分割后的图像转换为base64
            segments = []
            for i, img_array in enumerate(segmented_images):
                # 确保图像是RGB格式
                img_array = ensure_rgb_format(img_array)
                
                # 将numpy数组转换为PIL图像
                img = Image.fromarray(img_array)
                
                # 将图像转换为base64
                import base64
                from io import BytesIO
                
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                segments.append({
                    "image_base64": f"data:image/png;base64,{img_str}"
                })

            return jsonify({
                "success": True,
                "message": f"成功分割图像，检测到 {len(segments)} 个服装分割",
                "segments": segments
            })

        except Exception as e:
            logger.error(f"服装检测错误: {e}")
            return jsonify({
                "success": False,
                "message": f"服装检测错误: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"意外错误: {e}")
        return jsonify({
            "success": False,
            "message": f"意外错误: {str(e)}"
        }), 500


@api_up.route("/upload_image_fine_tuned", methods=["POST"])
@admin_required
@cross_origin()
def upload_image_fine_tuned():
    """
    前端上传单张图片的接口，使用微调后的模型处理，并保存到新的split_images表中

    请求格式:
    {
        "image_base64": "base64编码的图片字符串",
        "image_name": "可选的图片名称",
        "force_process": true/false (可选，是否强制处理即使没有检测到衣物)
    }

    返回格式:
    {
        "success": true/false,
        "message": "状态信息",
        "request_id": "请求ID，可用于取消请求",
        "data": {
            "original_image_id": "原图ID",
            "segments": [
                {
                    "splitted_image_id": "分割图片ID",
                    "splitted_image_path": "分割图片路径"
                },
                ...
            ]
        }
    }
    """
    start_time = time.time()

    # 生成唯一的请求ID
    request_id = str(uuid.uuid4())

    # 在追踪器中注册请求
    request_tracker.register_request(request_id)

    try:
        # 获取请求数据
        data = request.get_json()
        base64_image = data.get('image_base64')
        force_process = data.get('force_process', True)  # 默认强制处理

        # 如果未提供图片名称，则生成一个基于时间戳的唯一名称
        image_name = data.get('image_name', f"uploaded_{int(time.time())}")

        # 参数验证
        if not base64_image:
            logger.error("请求数据中缺少 'image_base64' 字段")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "缺少必要参数: 'image_base64'"
            }), 400

        logger.info(f"开始处理图片: {image_name}")

        # 步骤1: 将base64转换为图像数组
        try:
            convert_start = time.time()

            # 定义可取消的转换函数
            def convert_image():
                # 移除可能存在的数据前缀（例如 "data:image/jpeg;base64,"）
                if base64_image.startswith('data:'):
                    clean_base64 = base64_image.split(',', 1)[1]
                else:
                    clean_base64 = base64_image

                # 转换为numpy数组
                image_np = base64_to_numpy(clean_base64)

                # 确保图像是RGB格式（3通道）
                image_np = ensure_rgb_format(image_np)

                logger.info(f"图像转换用时: {time.time() - convert_start:.4f} 秒")
                logger.info(f"图像形状: {image_np.shape}")
                return image_np

            # 执行可取消操作
            image_np = request_tracker.run_cancellable(request_id, convert_image)

        except CancellationException:
            logger.info(f"请求 {request_id} 在图像转换过程中被取消")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "处理已被用户取消",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"图像转换错误: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"图像转换错误: {str(e)}",
                "request_id": request_id
            }), 400

        # 步骤2: 使用GroundingDINO检测服装物品
        try:
            detect_start = time.time()

            # 定义可取消的检测函数
            def detect_clothes():
                return clothing_detector.detect_clothes(image_np)

            # 执行可取消操作
            segmented_images = request_tracker.run_cancellable(request_id, detect_clothes)
            logger.info(f"服装检测用时: {time.time() - detect_start:.4f} 秒")

            # 检查是否检测到服装
            if not segmented_images and not force_process:
                request_tracker.complete_request(request_id)
                return jsonify({
                    "success": False,
                    "message": "未在图像中检测到服装物品",
                    "request_id": request_id
                }), 200

            # 如果没有检测到服装但强制处理，我们使用整个图像
            if not segmented_images and force_process:
                logger.warning("未检测到服装，但因强制处理标志而继续处理整张图像")
                # 将整张图像作为分割使用 - 确保是RGB格式
                segmented_images = [image_np.copy()]

        except CancellationException:
            logger.info(f"请求 {request_id} 在服装检测过程中被取消")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "处理已被用户取消",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"服装检测错误: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"服装检测错误: {str(e)}",
                "request_id": request_id
            }), 500

        # 步骤3: 创建数据目录存储分割图像
        data_dir = root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 为原图创建子目录
        image_dir = data_dir / image_name
        image_dir.mkdir(parents=True, exist_ok=True)

        # 步骤4: 处理并上传每个分割图像 - 使用微调模型提取特征
        uploaded_segments = []
        try:
            upload_start = time.time()

            for idx, img_array in enumerate(segmented_images):
                # 检查请求是否被取消
                if request_tracker.is_cancelled(request_id):
                    logger.info(f"请求 {request_id} 在图像上传过程中被取消")
                    request_tracker.complete_request(request_id)
                    return jsonify({
                        "success": False,
                        "message": "处理已被用户取消",
                        "request_id": request_id
                    }), 200

                # 再次确保每个分割图像是RGB格式
                img_array = ensure_rgb_format(img_array)

                # 输出当前处理的图像形状，用于调试
                logger.info(f"处理分割图像 {idx}，形状: {img_array.shape}")

                # 使用微调后的模型提取特征向量
                try:
                    feature_vector = ImageSimilarity.extract_feature(img_array)
                    # 转换为JSON字符串
                    vector_json = json.dumps(feature_vector.tolist())
                except Exception as e:
                    logger.error(f"特征提取错误: {e}")
                    vector_json = None

                filename = f"segment_{idx}.png"
                save_path = image_dir / filename

                # 保存图像到文件系统
                Image.fromarray(img_array).save(save_path)

                # 准备数据库存储信息
                splitted_image_id = f"{image_name}_segment_{idx}"
                relative_path = f"{image_name}/{filename}"
                
                # 将图像转换为二进制以便存储
                buffered = BytesIO()
                Image.fromarray(img_array).save(buffered, format="PNG")
                binary_data = buffered.getvalue()
                
                # 边界框信息 - 这里可以根据实际情况获取
                bounding_box = None
                
                # 保存到新的数据库表
                save_to_db(splitted_image_id, relative_path, image_name, bounding_box, vector_json, binary_data)
                
                uploaded_segments.append({
                    "splitted_image_id": splitted_image_id,
                    "splitted_image_path": relative_path
                })

            logger.info(f"图像上传用时: {time.time() - upload_start:.4f} 秒")

        except CancellationException:
            logger.info(f"请求 {request_id} 在分割图像上传过程中被取消")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": "处理已被用户取消",
                "request_id": request_id
            }), 200
        except Exception as e:
            logger.error(f"分割图像上传错误: {e}")
            request_tracker.complete_request(request_id)
            return jsonify({
                "success": False,
                "message": f"分割图像上传错误: {str(e)}",
                "request_id": request_id
            }), 500

        # 步骤5: 更新向量索引
        try:
            index_start = time.time()

            # 定义可取消的索引更新操作
            def update_index():
                return vector_index.rebuild_index()

            # 执行可取消操作
            index_result = request_tracker.run_cancellable(request_id, update_index)

            if index_result[0] is None:
                logger.warning("向量索引更新失败，但图像已成功上传")

            logger.info(f"索引更新用时: {time.time() - index_start:.4f} 秒")

        except CancellationException:
            logger.info(f"请求 {request_id} 在索引更新过程中被取消")
            # 注意: 即使索引更新被取消，我们仍然认为上传成功
            # 因为图像和向量数据已经保存到数据库中
        except Exception as e:
            logger.error(f"索引更新错误: {e}")
            # 仍然返回成功，只是带有警告信息

        # 标记请求为完成
        request_tracker.complete_request(request_id)

        total_time = time.time() - start_time
        logger.info(f"总处理时间: {total_time:.4f} 秒")

        return jsonify({
            "success": True,
            "message": f"成功处理并上传图像，检测到 {len(uploaded_segments)} 个服装分割",
            "request_id": request_id,
            "data": {
                "original_image_id": image_name,
                "segments": uploaded_segments
            }
        })

    except Exception as e:
        logger.error(f"意外错误: {e}")
        request_tracker.complete_request(request_id)
        return jsonify({
            "success": False,
            "message": f"意外错误: {str(e)}",
            "request_id": request_id
        }), 500