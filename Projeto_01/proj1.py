from skimage.io import imread
import skimage as sk
from skimage.feature import hog
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

class Aug:
	def __init__(self,filename):
		self.loc = "./datasets/PessoasICMC/p"
		self.filename = self.loc + str(filename) + "/" + str(filename)
		self.img = sk.io.imread(self.filename + ".png", as_gray = True)
		self.aug_img = []
		self.seed = 10

	def mirror(self, img):
		img_aux = np.flip(img, 1)
		return img_aux

	def rot(self, ang, img):
		img_aux = sk.transform.rotate(img, preserve_range = True, mode = 'constant', cval = 1, angle = ang)
		return img_aux

	def salt(self, img):
		img_aux = img
		img_aux = img_aux * sk.util.random_noise(img_aux, mode = 'pepper')
		return img_aux

	def aug_set(self):
		np.random.seed(self.seed)
		for i in range(1,10):
			aux_img = self.img
			if i % 2 == 0:
				aux_img = self.mirror(aux_img)
			aux_img = self.rot(np.random.randint(180), aux_img)
			aux_img = self.salt(aux_img)
			self.aug_img.append(aux_img.astype(np.uint16))
		self.save()

	def save(self):
		for i in range(1, 10):
			fname = self.filename + str(i) + ".png"
			sk.io.imsave(fname, self.aug_img[i-1])

class Hog:
	def __init__(self):
		self.icmc = []
		self.orl_faces = []

	
	def hog_ICMC(self):
		icmc_target = []
		path = "./datasets/PessoasICMC/p"
		for folder in range(1, 21):
			p = path + str(folder) + "/"
			
			for i in range(10):
				icmc_target.append(folder)
				if i == 0:
					img_name = p + str(folder) + ".png"
					
				else:
					img_name = p + str(folder) + str(i) + ".png"

				img = sk.io.imread(img_name, as_gray=True)

				hog_desc = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, multichannel=False, block_norm = "L1")
				#print(img_name + str(hog_desc))
				self.icmc.append(hog_desc)
		np.save("./datasets/icmc_target.npy", np.asarray(icmc_target))
		np.save("./datasets/icmc.npy", np.asarray(self.icmc))

	def hog_ORLFaces(self):
		path = "./datasets/OrlFaces20/s"
		orl_target = []
		for folder in range(1, 21):
			p = path + str(folder) + "/"
			
			for i in range(1, 11):
				orl_target.append(folder)
				img_name = p + str(i) + ".pgm"

				img = sk.io.imread(img_name, as_gray=True)

				hog_desc = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, multichannel=False, block_norm = "L1")
				#print(img_name + str(hog_desc))
				self.orl_faces.append(hog_desc)
		np.save("./datasets/orl_target.npy", np.asarray(orl_target))
		np.save("./datasets/orl.npy", np.asarray(self.orl_faces))

class KNN:
	def __init__(self, data_name):
		self.dataset = np.load("./datasets/" + data_name + ".npy")
		self.target = np.load("./datasets/" + data_name + "_target.npy")
		self.models = {}

	def knn_fold(self, id, k):
		self.models[str(id)] = {"class": KNeighborsClassifier(n_neighbors = k),
								"k": k,
								"score": 0
		}
		skf = StratifiedKFold(n_splits=10)
		scores = []
		for train_i, test_i in skf.split(self.dataset, self.target):
			self.models[str(id)]["class"].fit(self.dataset[train_i], self.target[train_i])
			score = self.models[str(id)]["class"].score(self.dataset[test_i], self.target[test_i])
			scores.append(score)
		self.models[str(id)]["score"] = np.mean(scores)
	
	def knn_test(self):
		self.knn_fold(0, 3)
		self.knn_fold(1, 5)
		self.knn_fold(2, 7)
		for c in self.models:
			print("KNN " + str(c) + "= " + str(self.models[c]["score"]))


	def confusion(self):
		classifier = KNeighborsClassifier(n_neighbors = 3)
		X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.target, test_size=0.2, random_state=42)
		classifier.fit(X_train, y_train)
		pred = classifier.predict(X_test)
		conf = confusion_matrix(y_test, pred)
		print(conf)
		#falta precisao e acuracia
	

class Perc:
	def __init__(self, data_name):
		self.dataset = np.load("./datasets/" + data_name + ".npy")
		self.target = np.load("./datasets/" + data_name + "_target.npy")
		self.models = {}

	def perc_fold(self,id, momentum, tamanho, aprendizado):
		
		self.models[str(id)] = {"class": MLPClassifier(hidden_layer_sizes=tamanho, learning_rate_init=aprendizado, momentum=momentum),
								"tamanho": tamanho,
								"momentum": momentum,
								"aprendizado": aprendizado,
								"score": 0
		}
		skf = StratifiedKFold(n_splits=10)
		scores = []
		for train_i, test_i in skf.split(self.dataset, self.target):
			self.models[str(id)]["class"].fit(self.dataset[train_i], self.target[train_i])
			score = self.models[str(id)]["class"].score(self.dataset[test_i], self.target[test_i])
			scores.append(score)
		self.models[str(id)]["score"] = np.mean(scores)

	def perc_test(self):
		self.perc_fold(0, 0.2, (500,), 0.5)
		self.perc_fold(1, 0.2, (500,), 0.9)
		self.perc_fold(2, 0.2, (500, 500,), 0.5)
		self.perc_fold(3, 0.2, (500, 500,), 0.9)
		self.perc_fold(4, 0.9, (500,), 0.5)
		self.perc_fold(5, 0.9, (500,), 0.9)
		self.perc_fold(6, 0.9, (500, 500,), 0.5)
		self.perc_fold(7, 0.9, (500, 500,), 0.9)
		for c in self.models:
			print("Perceptron " + str(c) + "= " + str(self.models[c]["score"]))


	def confusion(self):
		classifier = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.5, momentum=0.9)
		X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.target, test_size=0.2, random_state=42)
		classifier.fit(X_train, y_train)
		pred = classifier.predict(X_test)
		conf = confusion_matrix(y_test, pred)
		print(conf)
		#falta precisao e acuracia


class PrincCompAna:
	def __init__(self, data_name):
		self.dataset = np.load("./datasets/" + data_name + ".npy")
		self.data_name = data_name

	def apply_pca(self):
		p = PCA(n_components=0.5, svd_solver='full')
		reduced_dataset = p.fit_transform(self.dataset)
		pca_target = np.load("./datasets/" + self.data_name + "_target.npy")
		np.save("./datasets/" + self.data_name + "pca_target.npy", pca_target)
		np.save("./datasets/" + self.data_name + "pca.npy", np.asarray(reduced_dataset))


print("ORLFACES20")
data_name = "orl"
K = KNN(data_name)
P = Perc(data_name)
K.knn_test()
P.perc_test()
#print("KNN0 e Perceptron4 sao melhores")
#print("Matriz de confusao KNN0")
# K.confusion()
#print("Matriz de confusao Perceptron4")
# P.confusion()

print("ICMC")
data_name = "icmc"
K = KNN(data_name)
P = Perc(data_name)
K.knn_test()
P.perc_test()
# print("KNN0 e Perceptron4 sao melhores")
# print("Matriz de confusao KNN0")
# K.confusion()
# print("Matriz de confusao Perceptron4")
# P.confusion()

# print("ORLFACES20 PCA")
# data_name = "orlpca"
# K = KNN(data_name)
# P = Perc(data_name)
# print("Matriz de confusao KNN0")
# K.confusion()
# print("Matriz de confusao Perceptron4")
# P.confusion()

# print("ICMC PCA")
# data_name = "icmcpca"
# print("Matriz de confusao KNN0")
# K.confusion()
# print("Matriz de confusao Perceptron4")
# P.confusion()





