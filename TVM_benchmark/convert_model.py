import argparse
import os

import numpy as np
import torch

from models.layers import QConfig, QuantizeContext


def save_params(model, depth: int, save_path: str):
    # weight and bias (conv and dense)
    params = {}
    for (key, tensor) in model.items():
        if 'weight_integer' in key:
            print(key)
            params[key] = tensor.cpu().numpy().astype('int8')
        if 'bias_integer' in key:
            print(key)
            params[key] = tensor.cpu().numpy().astype('int32')

    renamed_params = {}
    renamed_params['embed_conv_weight'] = params['patch_embed.proj.weight_integer']
    renamed_params['embed_conv_bias'] = params['patch_embed.proj.bias_integer'].reshape(1, -1, 1, 1)

    for i in range(depth):
        for key in ['weight_integer', 'bias_integer']:
            old_name = f'blocks.{i}.attn.qkv.' + key
            new_name = f'block_{i}_attn_qkv_' + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = f'blocks.{i}.attn.proj.' + key
            new_name = f'block_{i}_attn_proj_' + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = f'blocks.{i}.mlp.fc1.' + key
            new_name = f'block_{i}_mlp_fc1_' + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = f'blocks.{i}.mlp.fc2.' + key
            new_name = f'block_{i}_mlp_fc2_' + key[:-8]
            renamed_params[new_name] = params[old_name]

    renamed_params['head_weight'] = params['head.weight_integer']
    renamed_params['head_bias'] = params['head.bias_integer']

    # norm
    for i in range(depth):
        for key in ['bias_integer']:
            old_name = f'blocks.{i}.norm1.' + key
            new_name = f'block_{i}_norm1_' + key[:-8]
            renamed_params[new_name] = model[old_name].cpu().numpy().astype('int32')

            old_name = f'blocks.{i}.norm2.' + key
            new_name = f'block_{i}_norm2_' + key[:-8]
            renamed_params[new_name] = model[old_name].cpu().numpy().astype('int32')

    renamed_params['norm_bias'] = model['norm.bias_integer'].cpu().numpy().astype('int32')

    # other params
    renamed_params['cls_token_weight'] = model['cls_token'].cpu().numpy()
    renamed_params['pos_embed_weight'] = model['pos_embed'].cpu().numpy()

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'params.npy'), renamed_params)


def load_qconfig(model, depth):
    params = {}
    for (key, tensor) in model.items():
        if 'scaling_factor' in key:
            tensor_np = tensor.cpu().numpy().reshape((-1))
            params[key] = tensor_np
        if 'act_scaling_factor' in key and np.ndim(tensor_np) == 1:
            tensor_np = tensor_np[0]
            params[key] = tensor_np

    QuantizeContext.qconfig_dict['qconfig_pos'] = QConfig(output_scale=params['qact_pos.act_scaling_factor'])
    QuantizeContext.qconfig_dict['qconfig_addpos'] = QConfig(input_scale=params['patch_embed.qact.act_scaling_factor'], input_dtype='int16', output_scale=params['qact1.act_scaling_factor'])
    # Embed
    conv_input_scale = params['qact_input.act_scaling_factor']
    conv_kernel_scale = params['patch_embed.proj.conv_scaling_factor']
    conv_output_scale = conv_input_scale * conv_kernel_scale
    QuantizeContext.qconfig_dict['qconfig_embed_conv'] = \
            QConfig(input_scale=conv_input_scale, kernel_scale=conv_kernel_scale, output_scale=conv_output_scale)

    for i in range(depth):
        input_scale = params['qact1.act_scaling_factor'] if i == 0 else params[f'blocks.{i - 1}.qact4.act_scaling_factor']
        output_scale = params[f'blocks.{i}.norm1.norm_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_norm1'] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.qact1.act_scaling_factor']
        kernel_scale = params[f'blocks.{i}.attn.qkv.fc_scaling_factor']
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_qkv'] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.attn.qact1.act_scaling_factor']
        output_scale = params[f'blocks.{i}.attn.matmul_1.act_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_matmul_1'] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.attn.qact_attn1.act_scaling_factor']
        output_scale = params[f'blocks.{i}.attn.int_softmax.act_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_softmax'] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.attn.int_softmax.act_scaling_factor']
        output_scale = params[f'blocks.{i}.attn.matmul_2.act_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_matmul_2'] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.attn.qact2.act_scaling_factor']
        kernel_scale = params[f'blocks.{i}.attn.proj.fc_scaling_factor']
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_proj'] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.attn.qact3.act_scaling_factor']
        output_scale = params[f'blocks.{i}.qact2.act_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_add1'] = QConfig(input_scale=input_scale, input_dtype='int16', output_scale=output_scale)

        input_scale = params[f'blocks.{i}.qact2.act_scaling_factor']
        output_scale = params[f'blocks.{i}.norm2.norm_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_norm2'] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.qact3.act_scaling_factor']
        kernel_scale = params[f'blocks.{i}.mlp.fc1.fc_scaling_factor']
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_fc1'] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.mlp.qact_gelu.act_scaling_factor']
        output_scale = params[f'blocks.{i}.mlp.act.act_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_gelu'] = QConfig(input_scale=input_scale, output_scale=output_scale, input_dtype='int8')

        input_scale = params[f'blocks.{i}.mlp.qact1.act_scaling_factor']
        kernel_scale = params[f'blocks.{i}.mlp.fc2.fc_scaling_factor']
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_fc2'] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params[f'blocks.{i}.mlp.qact2.act_scaling_factor']
        output_scale = params[f'blocks.{i}.qact4.act_scaling_factor']
        QuantizeContext.qconfig_dict[f'block_{i}_qconfig_add2'] = QConfig(input_scale=input_scale, input_dtype='int16', output_scale=output_scale)

    output_scale = params['norm.norm_scaling_factor']
    QuantizeContext.qconfig_dict['qconfig_norm'] = QConfig(input_scale=input_scale, output_scale=output_scale)

    input_scale = params['qact2.act_scaling_factor']
    kernel_scale = params['head.fc_scaling_factor']
    output_scale = input_scale * kernel_scale
    QuantizeContext.qconfig_dict['qconfig_head'] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='I-ViT convert model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'params_path', type=str, help='Saved parameters directory'
    )
    parser.add_argument(
        '--model-path', default='checkpoint.pth.tar', type=str,
        help='saved checkpoint path in QAT (checkpoint.pth.tar)'
    )
    parser.add_argument('--depth', default=12, type=int, help='Depth of ViT')

    args = parser.parse_args()
    model = torch.load(args.model_path)

    save_params(model, args.depth, args.params_path)
