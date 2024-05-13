import numpy as np
import random
import pickle

from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.data_utils import TinyImageNet, VerboseSubset, CIFAR

def get_iid_dataset(args):
	if args.dataset == 'cifar100':
		trasnform = SSLTransform(min_size=args.min_size, in_size=32, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
	elif args.dataset == 'tinyimagenet':
		transform = SSLTransform(min_size=args.min_size, in_size=64, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	dataset = datasets.ImageFolder(root=args.data_dir / 'train', transform=transform)
	return dataset

def get_stream_datasets(args):
	class_order = pickle.load(open(args.order_dir / 'class_order.pkl', 'rb'))
	if args.dataset == 'cifar100':
		dataset = CIFAR(args.data_dir, ordering=args.order, 
			transform=SSLTransform(min_size=args.min_size, in_size=32, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]), 
			num_tasks=args.num_tasks, class_order=class_order)
	elif args.dataset == 'tinyimagenet':
		dataset = TinyImageNet(args.order_dir, ordering=args.order, 
			transform=SSLTransform(min_size=args.min_size, in_size=64, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
			num_tasks=args.num_tasks, class_order=class_order)
	
	return dataset

def get_linear_eval_data_loaders(args):
	if args.dataset == 'cifar100':
		train_transform = transforms.Compose([
				transforms.RandomResizedCrop(32, scale=(args.min_size, 1), interpolation=Image.BICUBIC),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
									std=[0.2675, 0.2565, 0.2761])
		])
		test_transform = transforms.Compose([
				transforms.Resize(32, interpolation=Image.BICUBIC),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
									std=[0.2675, 0.2565, 0.2761])
		])
		trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
		testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)

	elif args.dataset == 'tinyimagenet':
		train_transform = transforms.Compose([
				transforms.RandomResizedCrop(64, scale=(args.min_size, 1), interpolation=Image.BICUBIC),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
		])
		test_transform = transforms.Compose([
				transforms.Resize(64, interpolation=Image.BICUBIC),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
		])
		trainset = datasets.ImageFolder(root=args.data_dir / 'train', transform=train_transform)
		testset = datasets.ImageFolder(root=args.data_dir / 'val', transform=test_transform)

	else:
		raise NotImplementedError

	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	return train_loader, test_loader

def get_knn_data_loaders(args):
	# No augmentation, no shuffle
	if args.dataset == 'tinyimagenet':
		test_transform = transforms.Compose([
				transforms.Resize(64, interpolation=Image.BICUBIC),
				#transforms.RandomGrayscale(p=1), #debug
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
		])

		trainset = datasets.ImageFolder(root=args.data_dir / 'train', transform=test_transform)
		testset = datasets.ImageFolder(root=args.data_dir / 'val', transform=test_transform)

	elif args.dataset == 'cifar100':
		test_transform = transforms.Compose([
				transforms.Resize(32, interpolation=Image.BICUBIC),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
									std=[0.2675, 0.2565, 0.2761])
		])
		trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=test_transform)
		testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)

	else:
		raise NotImplementedError

	if 'task' in args.order:
		train_loaders = []
		test_loaders = []
		class_order = pickle.load(open(args.order_dir / 'class_order.pkl', 'rb'))
		classes_per_task = len(class_order) // args.num_tasks
		stop = args.num_tasks
		for t_i in range(stop):
			class_names = class_order[t_i*classes_per_task:t_i*classes_per_task+classes_per_task]
			classes = [trainset.class_to_idx[c] for c in class_names]
			
			targets = np.array(trainset.targets)
			idx = (targets == classes[0])
			for class_idx in classes[1:]:
				idx |= (targets == class_idx)
			idx = [i for i, x in enumerate(idx) if x]
			task_trainset = VerboseSubset(trainset, idx, classes=classes)
			train_loaders.append(
				DataLoader(task_trainset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True))

			targets = np.array(testset.targets)
			idx = (targets == classes[0])
			for class_idx in classes[1:]:
				idx |= (targets == class_idx)
			idx = [i for i, x in enumerate(idx) if x]
			task_testset = VerboseSubset(testset, idx, classes=classes)
			test_loaders.append(
				DataLoader(task_testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True))

		return train_loaders, test_loaders

	else:
		train_loader = DataLoader(trainset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		test_loader = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

		return train_loader, test_loader



class GaussianBlur(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			sigma = random.random() * 1.9 + 0.1
			return img.filter(ImageFilter.GaussianBlur(sigma))
			#return transforms.functional.gaussian_blur(img, kernel_size=7, sigma=(0.1, 2.0))
		else:
			return img


class Solarization(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			return ImageOps.solarize(img)
		else:
			return img


class SSLTransform:
	def __init__(self, min_size, in_size, mean, std):
		self.transform1 = transforms.Compose([
			transforms.RandomResizedCrop(in_size, scale=(min_size, 1), interpolation=Image.BICUBIC),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=0.4, contrast=0.4,
										saturation=0.2, hue=0.1)],
				p=0.8
			),
			transforms.RandomGrayscale(p=0.2),
			GaussianBlur(p=1.0), 
			Solarization(p=0.0),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std)
		])
		self.transform2 = transforms.Compose([
			transforms.RandomResizedCrop(in_size, scale=(min_size, 1), interpolation=Image.BICUBIC),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=0.4, contrast=0.4,
										saturation=0.2, hue=0.1)],
				p=0.8
			),
			transforms.RandomGrayscale(p=0.2),
			GaussianBlur(p=0.1),
			Solarization(p=0.2),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std)
		])

	def __call__(self, x):
		y1 = self.transform1(x)
		y2 = self.transform2(x)
		return y1, y2


class BufferSSLTransform:
    """
    Adds ToPILImage()
    """
    def __init__(self, min_size, in_size, mean, std):
        self.transform1 = transforms.Compose([
        	transforms.ToPILImage(),
            transforms.RandomResizedCrop(in_size, scale=(min_size, 1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0), # lower?
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(in_size, scale=(min_size, 1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        y1 = self.transform1(x)
        y2 = self.transform2(x)
        return y1, y2
