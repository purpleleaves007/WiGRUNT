import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
#import cv2

class datatrcsi(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        count = 5400
        self.n_data = count

        for i in range(0,6):
            for j in range(1,10):
                for k in range(1,6):
                    for o in range(1,6):
                        for n in [2,3,4,5]:
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatecsi(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.n_data = 1350

        for i in range(0,6):
            for j in range(1,10):
                for k in range(1,6):
                    for o in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(1) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatrcsip(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        count = 3600
        self.n_data = count

        for i in range(0,6):
            for j in range(1,7):
                for k in [1,2,3,4]:
                    for o in range(1,6):
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatecsip(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.n_data = 900

        for i in range(0,6):
            for j in range(1,7):
                #for k in range(5):
                    for o in range(1,6):
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(5) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatrcsio(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        count = 4800
        self.n_data = count

        for i in range(0,6):
            for j in range(1,9):
                for k in range(1,6):
                    for o in [1,5,3,4]:
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatecsio(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.n_data = 1200

        for i in range(0,6):
            for j in range(1,9):
                for k in range(1,6):
                    #for o in range(5):
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(2) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatrcsie(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        count = 5250
        self.n_data = count

        for i in [9,10,11,12,13,14,15]: #0-8 E1, 9,10,15 E3 11-14 E2
            for j in range(1,7):
                for k in range(1,6):
                    for o in range(1,6):
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatecsie(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.n_data = 6750

        for i in [0,1,2,3,4,5,6,7,8]:
            for j in range(1,7):
                for k in range(1,6):
                    for o in range(1,6):
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data

class datatecsival(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.n_data = 1

        for i in [1]:
            for j in [3]:
                for k in [1]:
                    for o in [3]:
                        for n in [1]:
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), 
                                            ]) 

        img_paths, labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, labels

     def __len__(self):
        return self.n_data