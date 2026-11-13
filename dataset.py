import os
import random
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Scale(object):
    def __init__(self, size):
        self.size = (size, size)
    def __call__(self, input):
        input['img'] = input['img'].resize(self.size, Image.BILINEAR)
        input['co_gt'] = input['co_gt'].resize(self.size, Image.NEAREST)
        return input

class Random_Crop(object):
    def __init__(self, t_size):
        self.t_size = t_size
    def __call__(self, input):
        img = input['img']
        w, h = img.size
        if w > self.t_size and h > self.t_size:
            x1 = random.randint(0, w - self.t_size)
            y1 = random.randint(0, h - self.t_size)
            input['img'] = img.crop((x1, y1, x1 + self.t_size, y1 + self.t_size))
            input['co_gt'] = input['co_gt'].crop((x1, y1, x1 + self.t_size, y1 + self.t_size))
        return input

class Random_Flip(object):
    def __call__(self, input):
        if random.random() < 0.5:
           input['img'] = input['img'].transpose(Image.FLIP_LEFT_RIGHT)
           input['co_gt'] = input['co_gt'].transpose(Image.FLIP_LEFT_RIGHT)
        return input

class normalization(object):
    def __init__(self, split, scale_size=None):
        self.split = split
        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.ToTensor()
        elif self.split == 'test':
            self.img_transform = transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
        else:
            raise Exception("split not recognized")
    def __call__(self, input):
        input['img'] = self.img_transform(input['img'])
        input['co_gt'] = self.gt_transform(input['co_gt'])
        return input

class CoSOD_Train(data.Dataset):
    def __init__(self, args, split='train'):
        self.split = split
        self.train_data_root = args.train_data_root
        self.max_num = args.max_num
        self.size = args.img_size
        self.scale_size = args.scale_size

        # 加载所有数据集的路径
        train_datasets = args.train_datasets.split('+')
        self.all_imgs_dirs_list = []
        self.all_gts_dirs_list = []
        for dataset in train_datasets:
            root_path = os.path.join(self.train_data_root, dataset)
            imgs_list, gts_list = self.get_imgs_gts_dirs(root_path)
            self.all_imgs_dirs_list.extend(imgs_list)
            self.all_gts_dirs_list.extend(gts_list)
        print(f"总共加载了 {len(self.all_imgs_dirs_list)} 个图像组。")
        self._augmentation()

    def get_imgs_gts_dirs(self, root):
        """
        简化的加载函数，假设您的数据集结构为 root/img/类别/图片.png 和 root/gt/类别/标注.png
        """
        imgs_dirs_list = []
        gts_dirs_list = []
        img_dir = os.path.join(root, "img")
        gt_dir = os.path.join(root, "gt")
        # 如果找不到默认的 `img`/`gt`，尝试寻找 `images`/`gts`
        if not os.path.exists(img_dir):
            img_dir = os.path.join(root, "images")
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(root, "gts")
        if not (os.path.exists(img_dir) and os.path.exists(gt_dir)):
            raise ValueError(f"在 {root} 中找不到图片或标注文件夹 (尝试了 img/gt 和 images/gts)。")
        class_names = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        for cls in class_names:
            cls_img_dir = os.path.join(img_dir, cls)
            cls_gt_dir = os.path.join(gt_dir, cls)
            if not os.path.exists(cls_gt_dir):
                print(f"警告：跳过类别 {cls}，因为标注文件夹 {cls_gt_dir} 不存在。")
                continue
            img_files = sorted([f for f in os.listdir(cls_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not img_files:
                continue
            img_paths = [os.path.join(cls_img_dir, f) for f in img_files]
            # 构建标注路径，假设图片与标注文件名相同（扩展名可能为.png）
            gt_paths = []
            for img_f in img_files:
                base_name = os.path.splitext(img_f)[0]
                gt_f = base_name + '.png'  # 假设标注为png格式
                gt_path = os.path.join(cls_gt_dir, gt_f)
                if not os.path.exists(gt_path):
                    # 如果png不存在，尝试同名的jpg
                    gt_path = os.path.join(cls_gt_dir, base_name + '.jpg')
                if not os.path.exists(gt_path):
                    print(f"警告：未找到标注文件 {gt_f} 或 {base_name}.jpg，使用空标注。")
                    gt_path = ""  # 使用空字符串占位，您需要确保后续能处理
                gt_paths.append(gt_path)
            imgs_dirs_list.append(img_paths)
            gts_dirs_list.append(gt_paths)
        return imgs_dirs_list, gts_dirs_list

    def _augmentation(self):
        if self.split == 'train':
            self.joint_transform = Compose([Scale(self.scale_size), Random_Crop(self.size), Random_Flip()])
        elif self.split == 'test':
            self.joint_transform = None
        else:
            raise Exception("split not recognized")
        self.normalization = normalization(self.split, self.size)

    def __getitem__(self, item):
        imgs_path = self.all_imgs_dirs_list[item]
        co_gts_path = self.all_gts_dirs_list[item]
        num = len(imgs_path)
        if num > self.max_num:
            sample_list = random.sample(range(num), self.max_num)
            imgs_path = [imgs_path[i] for i in sample_list]
            co_gts_path = [co_gts_path[i] for i in sample_list]
            num = self.max_num
        imgs = torch.zeros(num, 3, self.size, self.size)
        co_gts = torch.zeros(num, 1, self.size, self.size)
        for idx in range(num):
            zip_data = {}
            img = Image.open(imgs_path[idx]).convert('RGB')
            # 处理可能为空的标注路径
            if co_gts_path[idx] and os.path.exists(co_gts_path[idx]):
                co_gt = Image.open(co_gts_path[idx]).convert('L')
            else:
                co_gt = Image.new('L', img.size, 0)  # 创建一个全黑的图作为标注
            zip_data['img'] = img
            zip_data['co_gt'] = co_gt
            if self.joint_transform:
                zip_data = self.joint_transform(zip_data)
            zip_data = self.normalization(zip_data)
            imgs[idx] = zip_data['img']
            co_gts[idx] = zip_data['co_gt']
        return {"imgs": imgs, "co_gts": co_gts}

    def __len__(self):
        return len(self.all_imgs_dirs_list)

class CoData_Test(data.Dataset):
    def __init__(self, img_root, img_size):
        # 查找 Image 文件夹，如果不存在则尝试 images
        image_dir = os.path.join(img_root, "Image")
        if not os.path.exists(image_dir):
            image_dir = os.path.join(img_root, "images")
        if not os.path.exists(image_dir):
            raise ValueError(f"在 {img_root} 中找不到 Image 或 images 文件夹。")
        
        class_list = os.listdir(image_dir)
        self.transform = normalization(split='test', scale_size=img_size)
        self.classes_dirs_list = list(
            map(lambda x: os.path.join(image_dir, x), class_list)
        )
        self.sizes = [img_size, img_size]

    def __getitem__(self, item):
        class_dir = self.classes_dirs_list[item]
        img_names = os.listdir(class_dir)
        num = len(img_names)
        img_paths = list(
            map(lambda x: os.path.join(class_dir, x), img_names)
        )
        
        # 查找 GroundTruth 文件夹，如果不存在则尝试 gt 或 gts
        possible_gt_dirs = ["GroundTruth", "gt", "gts"]
        gt_dir_base = None
        for dir_name in possible_gt_dirs:
            test_dir = os.path.join(os.path.dirname(os.path.dirname(class_dir)), dir_name)
            if os.path.exists(test_dir):
                gt_dir_base = test_dir
                break
        if gt_dir_base is None:
            raise ValueError(f"在数据根目录中找不到标注文件夹 (尝试了 {possible_gt_dirs})。")
        
        # 构建标注路径
        co_gt_paths = []
        for img_path in img_paths:
            class_name = os.path.basename(os.path.dirname(img_path))
            img_file_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 尝试多种标注文件扩展名
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                gt_path = os.path.join(gt_dir_base, class_name, img_file_name + ext)
                if os.path.exists(gt_path):
                    co_gt_paths.append(gt_path)
                    found = True
                    break
            
            if not found:
                # 如果找不到标注文件，创建一个空路径
                print(f"警告：未找到图片 {img_path} 对应的标注文件，将使用空标注。")
                co_gt_paths.append("")

        imgs = torch.zeros(num, 3, self.sizes[0], self.sizes[1])
        co_gts = torch.zeros(num, 1, self.sizes[0], self.sizes[1])

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            zip_data = {}
            img = Image.open(img_paths[idx]).convert('RGB')
            
            # 处理标注
            if co_gt_paths[idx] and os.path.exists(co_gt_paths[idx]):
                co_gt = Image.open(co_gt_paths[idx]).convert('L')
            else:
                co_gt = Image.new('L', img.size, 0)
            
            zip_data['img'] = img
            zip_data['co_gt'] = co_gt
            img_path_split = img_paths[idx].split('/')
            subpaths.append(
                os.path.join(
                    img_path_split[-2],
                    os.path.splitext(img_path_split[-1])[0] + '.png')
            )
            ori_sizes.append((img.size[1], img.size[0]))
            zip_data = self.transform(zip_data)
            imgs[idx] = zip_data["img"]
            co_gts[idx] = zip_data["co_gt"]

        return {
            "imgs": imgs,
            "co_gts": co_gts,
            "subpaths": subpaths,
            "ori_sizes": ori_sizes
        }

    def __len__(self):
        return len(self.classes_dirs_list)

def build_data_loader(args, mode):
    if mode == "train":
        train_dataset = CoSOD_Train(args, 'train')
        data_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        return data_loader
    elif mode == "test":
        test_root_dir = args.test_data_root
        test_datasets = args.test_datasets
        data_loaders = {}
        for dataset in test_datasets:
            data_root = os.path.join(test_root_dir, dataset)
            test_dataset = CoData_Test(
                img_root=data_root,
                img_size=args.img_size
            )
            data_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size
            )
            data_loaders[dataset] = data_loader
        return data_loaders
    else:
        raise RuntimeError