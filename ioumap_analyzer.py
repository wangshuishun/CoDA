import os
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DetectionResult:
    """存储检测结果的类"""
    image_name: str
    gt_area: float
    pred_area: float
    iou: float
    is_correct: bool = False

class StandardIOUCalculator:
    def __init__(self, data_root: str, result_dir: str = None):
        """
        初始化标准IOU计算器
        
        Args:
            data_root: 数据根目录路径（应包含gt和pred子目录）
            result_dir: 结果保存目录
        """
        self.data_root = Path(data_root)
        
        # 如果未指定结果目录，则创建在数据目录下的results文件夹
        if result_dir is None:
            self.result_dir = self.data_root / "results"
        else:
            self.result_dir = Path(result_dir)
            
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储结果
        self.all_results: List[DetectionResult] = []
        self.results_by_folder: Dict[str, List[DetectionResult]] = {}
        
    def find_largest_contour(self, binary_img: np.ndarray) -> np.ndarray:
        """
        找到图像中面积最大的连通域
        """
        if len(binary_img.shape) > 2:
            binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        return max(contours, key=cv2.contourArea)
    
    def get_min_area_rect(self, contour: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        计算最小外接矩形
        """
        if contour is None or len(contour) < 3:
            return 0.0, [(0, 0), (0, 0), (0, 0), (0, 0)]
            
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        area = cv2.contourArea(box)
        return area, box.tolist()
    
    def calculate_iou(self, rect1: List[Tuple[int, int]], rect2: List[Tuple[int, int]]) -> float:
        """
        计算两个旋转矩形的IOU
        """
        # 将矩形转换为轮廓
        rect1_contour = np.array(rect1, dtype=np.int32).reshape(-1, 1, 2)
        rect2_contour = np.array(rect2, dtype=np.int32).reshape(-1, 1, 2)
        
        # 计算交集
        intersection = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.drawContours(intersection, [rect1_contour], -1, 1, -1)
        intersection1 = intersection.copy()
        intersection = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.drawContours(intersection, [rect2_contour], -1, 1, -1)
        intersection2 = intersection.copy()
        
        # 计算交集和并集
        inter_area = np.logical_and(intersection1, intersection2).sum()
        union_area = np.logical_or(intersection1, intersection2).sum()
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_ap(self, ious: List[float], iou_threshold: float = 0.5) -> float:
        """
        计算单个IOU阈值下的Average Precision (AP)
        
        标准计算方法:
        1. 根据IOU判断每个预测是否正确
        2. 按置信度排序（这里我们假设所有预测置信度相同）
        3. 计算精确率-召回率曲线
        4. 计算曲线下面积
        
        Args:
            ious: 所有图像的IOU列表
            iou_threshold: IOU阈值
            
        Returns:
            AP值
        """
        if not ious:
            return 0.0
        
        # 判断每个预测是否正确
        correct = [1 if iou >= iou_threshold else 0 for iou in ious]
        
        # 由于我们没有置信度，假设所有预测按顺序排列
        # 在实际应用中，应该按置信度从高到低排序
        
        # 计算累计的TP和FP
        tp_cumsum = np.cumsum(correct)
        fp_cumsum = np.cumsum([1 - c for c in correct])
        
        # 计算精确率和召回率
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recalls = tp_cumsum / len(ious)
        
        # 添加(0, 0)和(1, 1)点用于插值
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])
        
        # 确保精确率单调递减（11点插值法）
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # 找到所有唯一的召回率点
        unique_recalls = np.unique(recalls)
        
        # 插值得到标准化的精确率值
        interpolated_precisions = []
        for recall_threshold in np.linspace(0, 1, 11):
            if recall_threshold in recalls:
                idx = np.where(recalls == recall_threshold)[0][-1]
                interpolated_precisions.append(precisions[idx])
            else:
                # 找到小于等于当前阈值的最大的召回率
                valid_recalls = recalls[recalls <= recall_threshold]
                if len(valid_recalls) > 0:
                    idx = np.where(recalls == valid_recalls[-1])[0][0]
                    interpolated_precisions.append(precisions[idx])
                else:
                    interpolated_precisions.append(0.0)
        
        # 计算AP（平均精确率）
        ap = np.mean(interpolated_precisions)
        return ap
    
    def calculate_map(self, iou_thresholds: List[float] = None) -> Dict[str, Any]:
        """
        计算标准的mAP (mean Average Precision)
        
        Args:
            iou_thresholds: IOU阈值列表，默认使用COCO的10个阈值
            
        Returns:
            包含mAP和各阈值AP的字典
        """
        if iou_thresholds is None:
            iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 
                            0.75, 0.80, 0.85, 0.90, 0.95]
        
        if not self.all_results:
            print("警告：没有可用的结果数据")
            return {}
        
        # 提取所有IOU值
        all_ious = [result.iou for result in self.all_results]
        
        # 计算每个阈值下的AP
        aps = []
        ap_details = {}
        
        for threshold in iou_thresholds:
            ap = self.calculate_ap(all_ious, threshold)
            aps.append(ap)
            ap_details[f'AP@{threshold:.2f}'] = float(ap)
        
        # 计算mAP（所有阈值AP的平均值）
        mean_ap = np.mean(aps) if aps else 0.0
        ap_details['mAP'] = float(mean_ap)
        
        # 计算IOU统计
        if all_ious:
            ap_details.update({
                'mean_iou': float(np.mean(all_ious)),
                'median_iou': float(np.median(all_ious)),
                'std_iou': float(np.std(all_ious)),
                'min_iou': float(np.min(all_ious)),
                'max_iou': float(np.max(all_ious)),
                'total_images': len(all_ious)
            })
        
        return ap_details
    
    def process_single_image(self, gt_path: Path, pred_path: Path, 
                           folder_name: str) -> DetectionResult:
        """
        处理单张图像，计算IOU
        """
        # 读取图像
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        
        if gt_img is None or pred_img is None:
            print(f"无法读取图像: {gt_path} 或 {pred_path}")
            return None
        
        # 找到最大连通域
        gt_contour = self.find_largest_contour(gt_img)
        pred_contour = self.find_largest_contour(pred_img)
        
        if gt_contour is None or pred_contour is None:
            print(f"图像中没有找到连通域: {gt_path} 或 {pred_path}")
            return None
        
        # 计算最小外接矩形
        gt_area, gt_rect = self.get_min_area_rect(gt_contour)
        pred_area, pred_rect = self.get_min_area_rect(pred_contour)
        
        # 计算IOU
        iou = self.calculate_iou(gt_rect, pred_rect)
        
        return DetectionResult(
            image_name=gt_path.name,
            gt_area=float(gt_area),
            pred_area=float(pred_area),
            iou=float(iou)
        )
    
    def process_folder(self, folder_name: str, gt_folder: Path, pred_folder: Path, visualize: bool = False):
        """
        处理单个文件夹（例如005）中的所有图像
        
        Args:
            folder_name: 文件夹名称（如'005'）
            gt_folder: 该文件夹对应的ground truth图像目录
            pred_folder: 该文件夹对应的预测图像目录
            visualize: 是否可视化（当前未使用）
        
        Returns:
            List[DetectionResult]: 该文件夹的结果列表
        """
        if not gt_folder.exists() or not pred_folder.exists():
            print(f"跳过 {folder_name}: 缺少gt或pred文件夹")
            return []
        
        # 获取图像文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        gt_files = []
        pred_files = []
        
        for ext in image_extensions:
            gt_files.extend(sorted(list(gt_folder.glob(ext))))
            pred_files.extend(sorted(list(pred_folder.glob(ext))))
        
        folder_results = []
        
        for gt_file in gt_files:
            # 寻找对应的预测文件（按文件名匹配）
            pred_file = None
            for pf in pred_files:
                if pf.name == gt_file.name:
                    pred_file = pf
                    break
            
            if pred_file is None:
                print(f"警告: 在 {folder_name} 中找不到与 {gt_file.name} 对应的pred文件")
                continue
            
            # 计算IOU
            result = self.process_single_image(gt_file, pred_file, folder_name)
            if result is not None:
                folder_results.append(result)
        
        return folder_results
    
    def process_all_folders(self, visualize: bool = False):
        """
        处理所有文件夹（gt和pred下的子目录）
        """
        # 构建gt和pred根目录
        gt_root = self.data_root / 'gt'
        pred_root = self.data_root / 'pred'
        
        if not gt_root.exists() or not pred_root.exists():
            raise FileNotFoundError(f"数据目录下未找到gt或pred文件夹: {self.data_root}")
        
        # 获取gt下的所有子文件夹（即各测试集名称，如005、006...）
        # 同时确保这些子文件夹在pred中也存在
        subfolders = []
        for d in gt_root.iterdir():
            if d.is_dir():
                # 检查pred下是否存在同名文件夹
                pred_sub = pred_root / d.name
                if pred_sub.exists() and pred_sub.is_dir():
                    subfolders.append(d.name)
                else:
                    print(f"警告: gt中存在文件夹 {d.name}，但pred中没有对应的文件夹，将跳过")
        
        subfolders = sorted(subfolders)
        print(f"开始处理 {len(subfolders)} 个文件夹...")
        
        all_results = []
        
        for folder_name in subfolders:
            print(f"\n处理文件夹: {folder_name}")
            gt_folder = gt_root / folder_name
            pred_folder = pred_root / folder_name
            
            folder_results = self.process_folder(folder_name, gt_folder, pred_folder, visualize)
            
            if folder_results:
                self.results_by_folder[folder_name] = folder_results
                all_results.extend(folder_results)
                
                # 输出文件夹统计
                iou_values = [r.iou for r in folder_results]
                print(f"  完成 {len(folder_results)} 张图像")
                print(f"  平均IOU: {np.mean(iou_values):.4f}")
                print(f"  IOU范围: {min(iou_values):.4f} - {max(iou_values):.4f}")
            else:
                print(f"  未处理任何图像")
        
        self.all_results = all_results
        
        # 输出总体统计
        if all_results:
            all_ious = [r.iou for r in all_results]
            print(f"\n所有文件夹总体统计:")
            print(f"  总图像数: {len(all_results)}")
            print(f"  总体平均IOU: {np.mean(all_ious):.4f}")
            print(f"  总体IOU范围: {min(all_ious):.4f} - {max(all_ious):.4f}")
        
        print("\n所有文件夹处理完成！")
    
    def save_detailed_results(self, output_path: Path):
        """
        保存详细的每张图片结果
        """
        detailed_results = {}
        
        for folder_name, results in self.results_by_folder.items():
            folder_data = []
            for result in results:
                folder_data.append({
                    'image_name': result.image_name,
                    'iou': result.iou,
                    'gt_area': result.gt_area,
                    'pred_area': result.pred_area,
                    'is_correct': result.is_correct
                })
            detailed_results[folder_name] = folder_data
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    def save_summary_report(self, output_path: Path, map_results: Dict):
        """
        保存汇总报告，包括每张图片的IOU
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("标准IOU和mAP评估报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"数据路径: {self.data_root}\n")
            f.write(f"结果保存路径: {self.result_dir}\n")
            f.write(f"处理时间: {np.datetime64('now')}\n\n")
            
            # 输出每张图片的详细结果
            f.write("各文件夹每张图片的详细结果:\n")
            f.write("=" * 70 + "\n")
            
            for folder_name, results in self.results_by_folder.items():
                f.write(f"\n文件夹: {folder_name}\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'图像名称':<30} {'IOU':<10} {'GT面积':<12} {'Pred面积':<12} {'状态':<10}\n")
                f.write("-" * 70 + "\n")
                
                for result in results:
                    status = "正确" if result.iou >= 0.5 else "错误"
                    f.write(f"{result.image_name:<30} {result.iou:<10.4f} "
                           f"{result.gt_area:<12.1f} {result.pred_area:<12.1f} "
                           f"{status:<10}\n")
                
                # 文件夹统计
                if results:
                    iou_values = [r.iou for r in results]
                    f.write("-" * 70 + "\n")
                    f.write(f"文件夹统计:\n")
                    f.write(f"  图像数量: {len(results)}\n")
                    f.write(f"  平均IOU: {np.mean(iou_values):.4f}\n")
                    f.write(f"  中位数IOU: {np.median(iou_values):.4f}\n")
                    f.write(f"  IOU范围: {min(iou_values):.4f} - {max(iou_values):.4f}\n")
                    f.write(f"  IOU标准差: {np.std(iou_values):.4f}\n\n")
            
            # 输出整体统计
            f.write("\n" + "=" * 70 + "\n")
            f.write("整体评估指标\n")
            f.write("=" * 70 + "\n\n")
            
            # IOU统计
            f.write("IOU统计:\n")
            f.write("-" * 70 + "\n")
            f.write(f"总图像数量: {map_results.get('total_images', 0)}\n")
            f.write(f"平均IOU: {map_results.get('mean_iou', 0):.4f}\n")
            f.write(f"中位数IOU: {map_results.get('median_iou', 0):.4f}\n")
            f.write(f"最小IOU: {map_results.get('min_iou', 0):.4f}\n")
            f.write(f"最大IOU: {map_results.get('max_iou', 0):.4f}\n")
            f.write(f"IOU标准差: {map_results.get('std_iou', 0):.4f}\n\n")
            
            # mAP统计
            f.write("mAP指标 (基于标准COCO评估方法):\n")
            f.write("-" * 70 + "\n")
            
            # 输出每个阈值下的AP
            for key, value in map_results.items():
                if key.startswith('AP@'):
                    f.write(f"{key}: {value:.4f}\n")
            
            f.write(f"\nmAP (所有阈值平均): {map_results.get('mAP', 0):.4f}\n")
            
            # 计算各阈值下的准确率
            if self.all_results:
                thresholds = [0.3, 0.5, 0.7, 0.9]
                f.write("\n不同IOU阈值下的准确率:\n")
                f.write("-" * 70 + "\n")
                for threshold in thresholds:
                    correct_count = sum(1 for r in self.all_results if r.iou >= threshold)
                    accuracy = correct_count / len(self.all_results)
                    f.write(f"IOU阈值 {threshold}: 准确率 = {accuracy:.4f} "
                           f"({correct_count}/{len(self.all_results)})\n")
    
    def visualize_iou_distribution(self, output_path: Path):
        """
        可视化IOU分布
        """
        if not self.all_results:
            return
        
        iou_values = [r.iou for r in self.all_results]
        
        plt.figure(figsize=(12, 8))
        
        # 1. IOU直方图
        plt.subplot(2, 2, 1)
        plt.hist(iou_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(iou_values), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(iou_values):.3f}')
        plt.axvline(np.median(iou_values), color='green', linestyle='--',
                   label=f'中位数: {np.median(iou_values):.3f}')
        plt.xlabel('IOU值')
        plt.ylabel('频数')
        plt.title('IOU分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. IOU箱线图（按文件夹）
        plt.subplot(2, 2, 2)
        if self.results_by_folder:
            folder_names = list(self.results_by_folder.keys())
            folder_iou_values = []
            for folder_name in folder_names:
                folder_iou = [r.iou for r in self.results_by_folder[folder_name]]
                folder_iou_values.append(folder_iou)
            
            box = plt.boxplot(folder_iou_values, labels=folder_names, patch_artist=True)
            for patch in box['boxes']:
                patch.set_facecolor('lightblue')
            plt.xlabel('文件夹')
            plt.ylabel('IOU值')
            plt.title('各文件夹IOU分布')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
        
        # 3. IOU累积分布图
        plt.subplot(2, 2, 3)
        sorted_ious = np.sort(iou_values)
        cdf = np.arange(1, len(sorted_ious) + 1) / len(sorted_ious)
        plt.plot(sorted_ious, cdf, linewidth=2)
        plt.fill_between(sorted_ious, 0, cdf, alpha=0.3)
        plt.xlabel('IOU值')
        plt.ylabel('累积概率')
        plt.title('IOU累积分布函数 (CDF)')
        plt.grid(True, alpha=0.3)
        
        # 4. IOU与面积关系图
        plt.subplot(2, 2, 4)
        gt_areas = [r.gt_area for r in self.all_results]
        iou_values_array = np.array(iou_values)
        
        scatter = plt.scatter(gt_areas, iou_values_array, alpha=0.6, 
                            c=iou_values_array, cmap='viridis')
        plt.colorbar(scatter, label='IOU值')
        plt.xlabel('Ground Truth面积')
        plt.ylabel('IOU值')
        plt.title('IOU与GT面积关系')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """
        保存所有结果
        """
        # 计算标准mAP
        map_results = self.calculate_map()
        
        # 保存详细结果
        detailed_path = self.result_dir / 'detailed_results.json'
        self.save_detailed_results(detailed_path)
        
        # 保存汇总报告（包含每张图片IOU）
        report_path = self.result_dir / 'summary_report.txt'
        self.save_summary_report(report_path, map_results)
        
        # 保存mAP结果
        map_path = self.result_dir / 'map_results.json'
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(map_results, f, indent=2, ensure_ascii=False)
        
        # 生成可视化图表
        viz_path = self.result_dir / 'iou_distribution.png'
        self.visualize_iou_distribution(viz_path)
        
        # 输出控制台摘要
        print("\n" + "=" * 70)
        print("评估结果摘要")
        print("=" * 70)
        
        if self.all_results:
            all_ious = [r.iou for r in self.all_results]
            print(f"\n总图像数量: {len(self.all_results)}")
            print(f"总体平均IOU: {np.mean(all_ious):.4f}")
            print(f"总体中位数IOU: {np.median(all_ious):.4f}")
            
            print(f"\n标准mAP指标:")
            print("-" * 70)
            print(f"mAP: {map_results.get('mAP', 0):.4f}")
            
            # 显示主要阈值下的AP
            for key in ['AP@0.50', 'AP@0.75', 'AP@0.95']:
                if key in map_results:
                    print(f"{key}: {map_results[key]:.4f}")
        
        print(f"\n结果文件:")
        print(f"  详细结果: {detailed_path}")
        print(f"  汇总报告: {report_path}")
        print(f"  mAP结果: {map_path}")
        print(f"  可视化图表: {viz_path}")
        print("=" * 70)

# 使用示例
if __name__ == "__main__":
    # 设置数据路径（应包含gt和pred两个子目录）
    data_root = "/workspace/WSY/CONDA-main/CONDA-main/ioumap/"
    result_dir = None  # 自动创建在data_root下的results文件夹
    
    print("开始计算标准IOU和mAP指标...")
    print(f"数据路径: {data_root}")
    
    # 创建计算器
    calculator = StandardIOUCalculator(data_root, result_dir)
    
    # 处理所有文件夹
    calculator.process_all_folders(visualize=False)
    
    # 保存结果
    calculator.save_results()