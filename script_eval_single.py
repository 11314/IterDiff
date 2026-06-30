print("begin")
import argparse
import os
from pprint import pprint
from typing import Literal

import torch
import torchvision.transforms.v2 as T
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import logging
from torchvision.io import read_image
from torchvision.utils import save_image

from attn_ctrl import register_attention_controller
from pipeline_emilie import EmiliePipeline
from pipeline_iter import IterEditPipeline
from pipeline_iterdiff import AttentionStore, IterDiffPipeline

logging.set_verbosity_error()
torch.enable_grad(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "/root/hxp/pretrained_models/instruct-pix2pix"

UNDERLYING_PIPE_TYPES = Literal["ip2p", "scfg", "iterdiff", "emilie"]   # 类型标注
UNDERLYING_PIPES = {    # 字典映射表，把命令行中输入的字符串，映射到具体的 Pipeline 类。
    "ip2p": StableDiffusionInstructPix2PixPipeline,
    "scfg": IterDiffPipeline,
    "iterdiff": IterDiffPipeline,
    "emilie": EmiliePipeline,
}


def load_single_image(image_path: str) -> torch.Tensor:
    """读取一张图像，并把它转换成模型可以直接使用的 tensor, 数值范围是 [0, 1]"""
    image = read_image(image_path)  # C, H, W; uint8

    # 确保输入图像是 3 通道 RGB 图像
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] >= 4:
        image = image[:3]

    transform = T.Compose(  # 定义图像预处理操作
        [
            # For face editing, using a fixed 512x512 input is usually safer.
            T.Resize((512, 512), antialias=True),   # 统一缩放
            T.ToDtype(torch.float32, scale=True),   # 转成float32，并归一化
        ]
    )
    image = transform(image)    # 进行预处理
    return image.unsqueeze(0)  # 1, C, H, W


def build_iter_prompts(prompts: list[str]) -> list[list[str]]:
    """
    把 prompt 转换成原始 benchmark dataloader 使用的格式

    Original batch_size=1 format:
        [[step1_prompt], [step2_prompt], [step3_prompt], ...]
    """
    return [[p] for p in prompts]


def get_iterpipe(underlying_pipe_type: UNDERLYING_PIPE_TYPES) -> IterEditPipeline:  # 根据传入的type加载对应的pipeline
    underlying_pipe_cls = UNDERLYING_PIPES[underlying_pipe_type]

    return IterEditPipeline(
        underlying_pipe_cls.from_pretrained(    # 加载预训练模型
            MODEL_ID, torch_dtype=torch.float32, safety_checker=None
        )
    ).to(DEVICE)


def save_result(
    results_dir: str,
    images: torch.Tensor,
    results: torch.Tensor,
    title: str = "",
    filename: str = "single",
):
    save_dir = os.path.join(results_dir, title, filename)
    os.makedirs(save_dir, exist_ok=True)

    for orig_image, edited_images in zip(images.unbind(dim=0), results.unbind(dim=0)):
        save_image(orig_image, os.path.join(save_dir, "0.png")) # 原始图像保存为0

        for i, img in enumerate(edited_images, start=1):    # 编辑后的图像从1开始编号
            save_image(img, os.path.join(save_dir, f"{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=UNDERLYING_PIPES.keys(), required=True)

    parser.add_argument("--mb_size", type=int)
    parser.add_argument("--mb_save_topk", type=int, default=20)
    parser.add_argument("--use_factor", action="store_true")

    parser.add_argument("--exp_title", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True, help="Path to one input image")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        required=True,
        help="One or more edit instructions. Quote each instruction separately.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to save the results",
    )
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    print("Running with the following arguments:")
    print(args)

    image = load_single_image(args.image_path).to(DEVICE)   # 加载并预处理图像
    insts = build_iter_prompts(args.prompt) # 构造迭代编辑 prompt 格式

    pipe = get_iterpipe(args.type)  # 加载编辑管道

    if args.type == "scfg": # 不同 type 设置额外参数
        controller = AttentionStore(
            size=pipe.pipe.unet.sample_size,
            mb_size=0,
            mb_save_topk=0,
        )
        register_attention_controller(pipe.pipe.unet, controller)

        pipe_kwargs = {
            "attn_ctrl": controller,
            "use_scfg": True,
            "use_factor": False,
        }
    elif args.type == "iterdiff":   # 默认使用这个
        if args.mb_size is None:
            raise ValueError("`--mb_size` is required when `--type iterdiff`.")

        controller = AttentionStore(
            size=pipe.pipe.unet.sample_size,
            mb_size=args.mb_size,
            mb_save_topk=args.mb_save_topk,
        )
        register_attention_controller(pipe.pipe.unet, controller)

        pipe_kwargs = {
            "attn_ctrl": controller,
            "use_scfg": True,
            "use_factor": args.use_factor,
        }
    else:  # args.type in ["ip2p", "emilie"]
        pipe_kwargs = {}

    if "attn_ctrl" in pipe_kwargs:  # 重置注意力控制器状态
        pipe_kwargs["attn_ctrl"].full_reset()

    results = pipe( # 正式执行编辑
        prompt=insts,
        image=image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        generator=torch.Generator(device=DEVICE).manual_seed(args.seed),
        **pipe_kwargs,
    )

    if args.type == "emilie":
        pipe.pipe.clear_cache()

    save_result(args.results_dir, image, results, args.exp_title, "single") # 保存编辑结果
