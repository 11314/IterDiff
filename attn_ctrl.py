import abc

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class AttentionControl(abc.ABC):
    def __init__(self):
        self.num_cross_att_layers = 0
        self.num_self_att_layers = 0
        self.cur_step = 0
        self.cur_attn_layer = 0
        self.cur_edit = 0

    def init_att_layers_count(self, unet: UNet2DConditionModel):
        for m in unet.modules():    # 遍历unet中的所有子模块。
            if isinstance(m, Attention):    # 跳过普通卷积层、归一化层、激活层，只关注 attention 层。
                if m.is_cross_attention:
                    self.num_cross_att_layers += 1
                else:
                    self.num_self_att_layers += 1

    def reset(self):
        self.cur_step = 0
        self.cur_attn_layer = 0

    def step_callback(self, *args, **kwargs):
        return

    def between_steps(self):    # 预留的接口，当整个unet遍历之后，可以执行某些操作
        return

    def between_edits(self):
        self.reset()
        self.cur_edit += 1

    @property
    def num_att_layers(self):   # 判断是否统计到任何注意力层，如果没有就进入
        if self.num_cross_att_layers == 0 and self.num_self_att_layers == 0:
            logger.warning(
                "No attention layers found in the UNet."
                f"Please call `{self.__class__.__name__}.init_att_layers_count` or set `num_cross_att_layers` and `num_self_att_layers` manually."
            )
        return self.num_cross_att_layers + self.num_self_att_layers

    @abc.abstractmethod
    def forward(
        self,
        tensors: dict[str, torch.Tensor],
        is_cross: bool,
        attn_processor_name: str,
    ):
        raise NotImplementedError

    def __call__(
        self,
        tensors: dict[str, torch.Tensor],
        is_cross: bool,
        attn_processor_name: str,
    ):
        return self.forward(tensors, is_cross, attn_processor_name)

    def next_attn_layer(self):  # 扩散步数的计数
        if self.cur_attn_layer < self.num_att_layers:  # inside the unet
            self.cur_attn_layer += 1
        if self.cur_attn_layer == self.num_att_layers:  # after the unet
            self.between_steps()
            self.cur_attn_layer = 0
            self.cur_step += 1


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, attn_processor_name: str, **kwargs):    # 返回输入的原注意力
        return attn


class AttnControlProcessor:
    def __init__(self, attn_ctrl: AttentionControl, attn_processor_name: str):
        self.attn_ctrl = attn_ctrl
        self.attn_processor_name = attn_processor_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states    #  保存残差连接的输入

        if attn.spatial_norm is not None:   # 如果有空间注意力，就做归一化
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim # 记录输入维度向量

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view( # 展平空间维度
                batch_size, channel, height * width
            ).transpose(1, 2)   # 转置维度

        batch_size, sequence_length, _ = (  # 获取 batch size 和 sequence length
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(   # 准备mask
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None: # 如果有group_norm，就做归一化
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:   # 没有传入 encoder_hidden_states，说明当前是自注意力。
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)  # 计算K,V
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)   # 多头合并
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        key, value = self.attn_ctrl(
            tensors={"key": key, "value": value},
            is_cross=attn.is_cross_attention,
            attn_processor_name=self.attn_processor_name,
        )

        attention_probs = attn.get_attention_scores(query, key, attention_mask) # 计算注意力权重

        attention_probs = self.attn_ctrl(   # 调用注意力传感器，处理权重
            tensors={"attn": attention_probs},
            is_cross=attn.is_cross_attention,
            attn_processor_name=self.attn_processor_name,
        )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.attn_ctrl.next_attn_layer()    # 更新attention层计数

        return hidden_states


def register_attention_controller(  # 将自定义的AttnControlProcessor挂钩到注意力层中
    unet: UNet2DConditionModel, controller: AttentionControl
):
    attn_processors = {
        name: AttnControlProcessor(attn_ctrl=controller, attn_processor_name=name)
        for name in unet.attn_processors.keys() # 取出 U-Net 中所有 attention processor 的名字
    }

    unet.set_attn_processor(attn_processors)  # 替换attention_processors
    controller.init_att_layers_count(unet)  # 统计所有attention_processors的数量
