import os
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from config.config import get_cfg
from dataset import build_data_loader

from CoSODNet import CoSODNet
import transforms as trans


def get_args_parser():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser("CoSOD_Test", add_help=False)
    parser.add_argument("-config_file", default="./config/cosod.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("-num_works", default=1, type=int)
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-device_id", type=str, default="0")
    parser.add_argument("-img_size", type=int, default=256)
    parser.add_argument("-max_num", type=int, default=13)
    parser.add_argument("-model_path", type=str, 
                        default="/workspace/WSY/CONDA-main/CONDA-main/models/MyTest/iterations33000.pth",
                        help="path to your trained model weights")
    parser.add_argument("-test_data_root", type=str, default="./dataset/test_data")
    parser.add_argument("-test_datasets", nargs='+', default=["CoCA", "CoSal2015", "CoSOD3k"])
    parser.add_argument("-save_dir", type=str, default='./predictions')
    parser.add_argument("-model_name", type=str, default="MyTest_iter33000",
                        help="name for saving results")
    return parser


def _get_cfg(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    return cfg


def test_group(model, group_data, save_root, max_num, device):
    img_num = group_data['imgs'].shape[1]
    groups = list(range(0, img_num + 1, max_num))
    if groups[-1] != img_num:
        groups.append(img_num)

    print(f"分组索引: {groups}")

    for i in range(len(groups) - 1):
        if i == len(groups) - 2:
            end = groups[i + 1]
            start = max(0, end - max_num)
        else:
            start = groups[i]
            end = groups[i + 1]

        print(f"处理第 {i+1}/{len(groups)-1} 组: 起始 {start}, 结束 {end}")

        inputs = Variable(group_data['imgs'][:, start:end].squeeze(0).to(device))
        subpaths = group_data['subpaths'][start:end]
        ori_sizes = group_data['ori_sizes'][start:end]

        with torch.no_grad():
            result = model(inputs)
            pred_prob = torch.sigmoid(result['final_pred'])

            for p_id in range(end - start):
                pre = pred_prob[p_id, :, :, :].data.cpu()

                subpath = subpaths[p_id][0]
                ori_size = (ori_sizes[p_id][1].item(),
                            ori_sizes[p_id][0].item())

                transform = trans.Compose([
                    trans.ToPILImage(),
                    trans.Scale(ori_size)
                ])
                outputImage = transform(pre)
                
                # 解析原始路径结构，保持相同的目录层次
                # 假设 subpath 是类似 "005/img1.jpg" 的形式
                # 我们要在 save_root 下创建相同的子目录结构
                
                # 分割路径
                path_parts = subpath.split('/')
                
                # 确保至少有一个目录和一个文件名
                if len(path_parts) >= 2:
                    # 重建保存路径：save_root + 原目录结构
                    category_dir = os.path.join(save_root, path_parts[0])
                    os.makedirs(category_dir, exist_ok=True)
                    
                    # 获取原始文件名（不包括路径）
                    original_filename = path_parts[-1]
                    
                    # 生成保存路径
                    # 如果需要可以修改扩展名，比如从 .jpg 改为 .png
                    # 这里保持原文件名，但使用 .png 扩展名
                    save_filename = os.path.splitext(original_filename)[0] + '.png'
                    save_path = os.path.join(category_dir, save_filename)
                    
                    outputImage.save(save_path)
                    print(f"保存预测图到: {save_path}")
                else:
                    # 如果没有目录层级，直接保存在 save_root 下
                    filename = os.path.basename(subpath)
                    save_filename = os.path.splitext(filename)[0] + '.png'
                    save_path = os.path.join(save_root, save_filename)
                    outputImage.save(save_path)
                    print(f"保存预测图到: {save_path}")


def main(args):
    # 设置设备（与train.py一致）
    if torch.cuda.is_available():
        # 根据 device_id 设置设备
        if args.device_id:
            # 如果有多个设备ID，使用第一个
            device_id = args.device_id.split(',')[0] if ',' in args.device_id else args.device_id
            device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(int(device_id))
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 加载配置
    cfg = _get_cfg(args.config_file)
    
    # 创建模型
    model = CoSODNet(cfg)
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        print("请确保路径正确，或使用 -model_path 参数指定正确的路径")
        return
    
    print(f"加载模型: {args.model_path}")
    try:
        # 加载模型权重
        # 使用 map_location 参数确保权重加载到正确的设备
        state_dict = torch.load(args.model_path, map_location=device)
        
        # 检查加载的 state_dict 格式
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # 如果是完整的检查点（包含优化器等）
            model.load_state_dict(state_dict['model_state_dict'])
            print("从完整检查点加载模型权重")
        else:
            # 如果是纯模型权重
            model.load_state_dict(state_dict)
            print("加载模型权重成功")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("可能是模型文件格式不匹配或路径错误")
        return
    
    # 构建测试数据加载器
    print("构建数据加载器...")
    test_loaders = build_data_loader(args, mode='test')
    
    # 检查是否有测试数据加载器
    if not test_loaders:
        print("错误: 无法构建测试数据加载器")
        return
    
    # 使用指定的模型名称或从路径提取
    if args.model_name:
        model_save_name = args.model_name
    else:
        # 从路径提取模型名称
        model_save_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    print(f"使用模型名称: {model_save_name}")
    
    # 测试所有指定的数据集
    for dataset in args.test_datasets:
        if dataset not in test_loaders:
            print(f"警告: 数据集 {dataset} 不可用，跳过")
            print(f"可用的数据集: {list(test_loaders.keys())}")
            continue
        
        data_loader = test_loaders[dataset]
        print(f"测试数据集: {dataset}，包含 {len(data_loader)} 个图像组")
        
        # 创建保存目录
        save_root = os.path.join(args.save_dir, dataset, model_save_name)
        os.makedirs(save_root, exist_ok=True)
        print(f"结果将保存到: {save_root}")
        
        for idx, group_data in enumerate(data_loader):
            print(f'处理第 {idx+1}/{len(data_loader)} 组')
            
            max_num = args.max_num
            success = False
            
            # 重试机制，如果内存不足则减少批次大小
            while not success:
                try:
                    test_group(model, group_data, save_root, max_num, device)
                    success = True
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"内存不足，将 max_num 从 {max_num} 减少到 {max_num-1}")
                        max_num = max_num - 1
                        if max_num <= 0:
                            print("错误: max_num 已减少到 0，无法继续处理")
                            success = True  # 跳出循环，避免无限循环
                    else:
                        print(f"处理第 {idx+1} 组时出错: {e}")
                        success = True  # 跳出循环，避免无限循环
                except Exception as e:
                    print(f"处理第 {idx+1} 组时出现未知错误: {e}")
                    success = True  # 跳出循环，避免无限循环
        
        print(f"数据集 {dataset} 测试完成! 结果保存在: {save_root}")
        
        # 展示生成的目录结构
        print("\n生成的目录结构:")
        for root, dirs, files in os.walk(save_root):
            level = root.replace(save_root, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for file in files[:5]:  # 只显示前5个文件
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... 还有 {len(files)-5} 个文件')
        
        print("-" * 50)
    
    print("所有数据集测试完成!")


if __name__ == '__main__':
    ap = argparse.ArgumentParser("CoSOD testing script", parents=[get_args_parser()])
    args = ap.parse_args()
    
    # 启用cudnn加速
    cudnn.benchmark = True
    
    # 检查必要的目录是否存在
    if not os.path.exists(args.test_data_root):
        print(f"警告: 测试数据目录 {args.test_data_root} 不存在")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"警告: 模型文件 {args.model_path} 不存在")
        print("请使用 -model_path 参数指定正确的模型路径")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 运行主函数
    main(args)