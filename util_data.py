import cv2
import torch
import os
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])

class DataSet():
	def __init__(self, N, spt, qry, file_path = './images_evaluation', re_size=False, transform=None):
		self.N = N
		self.spt = spt
		self.qry = qry
		self.file_path = file_path
		self.re_size = re_size
		self.transform = transform
		self.class_set = self.data_generator()
		self.train_data = self.get_dataset('spt')
		self.test_data = self.get_dataset('qry')

	def get_dataset(self, train_or_test):
		data_list = []
		for idx in range(len(self.class_set[train_or_test])):
			image = self.class_set[train_or_test][idx]
			if self.transform:
				image = self.transform(image)
			data_list.append((image, self.class_set['label'][idx]))
		return data_list

	def data_generator(self):
		# Read images data from local
		contents = os.listdir(self.file_path)
		label = 0
		label_list = []
		ori_image_dataset = []
		for set in contents:
			cur_dir = self.file_path + '/' + set
			for item in os.listdir(cur_dir):
				label_list.append(label)
				single_class_image = []
				for image in os.listdir(cur_dir + '/' + item):
					new_img = cv2.imread(f'{cur_dir}/{item}/{image}', cv2.IMREAD_GRAYSCALE)
					if self.re_size == True:
						new_img = cv2.resize(new_img, dsize=(28, 28))
					new_img = np.array(new_img, dtype=np.float)
					single_class_image.append(new_img)
				label += 1
				ori_image_dataset.append(single_class_image)

		# Sampler
		random.shuffle(label_list)
		label_list = label_list[: self.N]
		# One-hot label
		one_hot_index = list(np.arange(self.N))
		one_hot_index = np.identity(self.N)[one_hot_index]
		# Spt And Qry
		spt_set = []
		qry_set = []
		one_hot_label = []
		label_0 = 0
		for idx in label_list:
			spt_select_list = random.sample(ori_image_dataset[idx][:11], self.spt)
			qry_select_list = random.sample(ori_image_dataset[idx][11:], self.qry)
			for item1 in spt_select_list:
				one_hot_label.append(np.array(one_hot_index[label_0], dtype=np.float))
				spt_set.append(item1)
			for item2 in qry_select_list:
				qry_set.append(item2)
			label_0 += 1
		N_class_set = {'spt': spt_set, 'qry': qry_set, 'label': one_hot_label}
		return N_class_set

	def get_train_data(self):
		return self.train_data

	def get_test_data(self):
		return self.test_data


if __name__ == '__main__':
	data_set = DataSet(5, 5, 5, re_size=True, transform=data_transform)
	train_dataset = data_set.get_train_data()
	test_dataset = data_set.get_test_data()
	image, target = train_dataset[0]
	# print(image.shape, target)
	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
	for data in test_loader:
		imgs, targets = data
		print(targets)
