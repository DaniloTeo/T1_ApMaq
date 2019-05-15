# Reconhecimento de Faces
# Aprendizado de Máquina - SCC0276
# Danilo da Costa Telles Téo	9293626
# Rodrigo Valim Maciel			9278149


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

# Classe para Data Augmentation da Base de Dados de pessoas do ICMC. São aplicados variados métodos para alterar a imagem
# fornecida, focando em permutações de associações de aplicação de ruídos e rotação da imagem. 
# Salva em disco na pasta da pessoa 'aumentada' e com um indice de [1,9] como apendice. Ex.: p20/203.png
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


# Classe para fazer extração dos HOGs de cada uma das bases de dados de maneira organizada e salvar em disco.	
# Salva os HOGs de cada base dentro da pasta 'datasets'. Para cada base são gerados dois arrays numpy, um contendo
# os HOGs das imagens alvo (Ex.: icmc_target.npy) e outro contendo o restante das imagens (Ex.: icmc.npy).
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


# Aplica o PCA para uma base de dados, cujo filename é fornecido por argumento. São salvos em disco dois array numpy, um
# para os target (pca_target.npy) e outro para o restante dos HOGs (pca.npy)
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


# Configura diversos classificadores KNN. Realiza o treinamento e o teste, utilizando 10-fold validation,
# associando cada score a seu devido classificador, organizados em uma lista de Dictionaries
class KNN:
	def __init__(self, data_name):
		self.dataset = np.load("./datasets/" + data_name + ".npy")
		self.target = np.load("./datasets/" + data_name + "_target.npy")
		self.models = []

	def knn_fold(self, id, k):
		self.models.append({"class": KNeighborsClassifier(n_neighbors = k),
										"k": k,
										"score": 0,
										"id": id})
		skf = StratifiedKFold(n_splits=10)
		scores = []
		for train_i, test_i in skf.split(self.dataset, self.target):
			self.models[id]["class"].fit(self.dataset[train_i], self.target[train_i])
			score = self.models[id]["class"].score(self.dataset[test_i], self.target[test_i])
			scores.append(score)
		self.models[id]["score"] = np.mean(scores)
	
	def knn_test(self):
		self.knn_fold(0, 3)
		self.knn_fold(1, 5)
		self.knn_fold(2, 7)
		for c in range(len(self.models)):
			print("KNN " + str(c) + "= " + str(self.models[c]["score"]))

	def get_best_knn(self):
		aux = sorted(self.models, key=lambda k: k['score'], reverse=True)
		return aux[0]


	def confusion(self):
		classifier = self.get_best_knn()
		X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.target, test_size=0.2, random_state=42)
		classifier['class'].fit(X_train, y_train)
		pred = classifier['class'].predict(X_test)
		conf = Confusion(y_test, pred).con_mat
		precisions = np.zeros([len(conf),])
		acc = np.trace(conf)/np.sum(conf)
		for line in range(len(conf)):
			if np.sum(conf[line]) != 0:
				precisions[line] = conf[line][line]/(np.sum(conf[line]))
			else:
				precisions[line] = 0

		print("Matriz de confusao KNN " + str(classifier['id']))
		print(conf)
		print("Acuracia: " + str(acc))
		print("Precisoes por Classe:")
		for i in range(len(precisions)):
			print("\tClasse " + str(i) + ": " + str(precisions[i]))
	

# Configura diversos classificadores Multilayer Perceptron. Realiza o treinamento e o teste, utilizando 10-fold validation,
# associando cada score a seu devido classificador, organizados em uma lista de Dictionaries
class Perc:
	def __init__(self, data_name):
		self.dataset = np.load("./datasets/" + data_name + ".npy")
		self.target = np.load("./datasets/" + data_name + "_target.npy")
		self.models = []

	def perc_fold(self,id, momentum, tamanho, aprendizado):
		
		self.models.append({"class": MLPClassifier(hidden_layer_sizes=tamanho, learning_rate_init=aprendizado, momentum=momentum),
										"tamanho": tamanho,
										"momentum": momentum,
										"aprendizado": aprendizado,
										"score": 0,
										"id": id})
		skf = StratifiedKFold(n_splits=10)
		scores = []
		for train_i, test_i in skf.split(self.dataset, self.target):
			self.models[id]["class"].fit(self.dataset[train_i], self.target[train_i])
			score = self.models[id]["class"].score(self.dataset[test_i], self.target[test_i])
			scores.append(score)
		self.models[id]["score"] = np.mean(scores)

	def perc_test(self):
		self.perc_fold(0, 0.2, (500,), 0.5)
		self.perc_fold(1, 0.2, (500,), 0.9)
		self.perc_fold(2, 0.2, (500, 500,), 0.5)
		self.perc_fold(3, 0.2, (500, 500,), 0.9)
		self.perc_fold(4, 0.9, (500,), 0.5)
		self.perc_fold(5, 0.9, (500,), 0.9)
		self.perc_fold(6, 0.9, (500, 500,), 0.5)
		self.perc_fold(7, 0.9, (500, 500,), 0.9)
		for c in range(len(self.models)):
			print("Perceptron " + str(c) + "= " + str(self.models[c]['score']))


	def get_best_perc(self):
		aux = sorted(self.models, key=lambda k: k['score'], reverse=True)
		return aux[0]

	def confusion(self):
		classifier = self.get_best_perc()
		X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.target, test_size=0.2, random_state=42)
		classifier['class'].fit(X_train, y_train)
		pred = classifier['class'].predict(X_test)
		conf = Confusion(y_test, pred).con_mat
		precisions = np.zeros([len(conf),])
		acc = np.trace(conf)/np.sum(conf)
		for line in range(len(conf)):
			if np.sum(conf[line]) != 0:
				precisions[line] = conf[line][line]/(np.sum(conf[line]))
			else:
				precisions[line] = 0

		print("Matriz de confusao Perceptron " + str(classifier['id']))
		print(conf)
		print("Acuracia: " + str(acc))
		print("Precisoes por Classe:")
		for i in range(len(precisions)):
			print("\tClasse " + str(i) + ": " + str(precisions[i]))

class Confusion:
	def __init__(self, y_test, pred):
		self.y_test = y_test
		self.pred = pred
		self.con_mat = self.fill()

	def fill(self):
		mat = np.zeros([20,20]).astype(int)
		self.y_test = self.y_test - 1
		self.pred = self.pred - 1
		for i in range(len(self.y_test)):
			mat[self.y_test[i]][self.pred[i]] += 1
		return mat	



# MAIN:
#---------------------------------------------------------------------------------------------------------------------
#Funções de execução única. Uma vez que este grupo de operações é executado ele não precisa ser executados para
#os experimentos seguintes.


#Data Augmentation da Base de Dados de Pessoas do ICMC, e escrita das imagens geradas no disco
for i in range(1,21):
	A = Aug(i)
	A.aug_set()

# Criação dos Histograms of Oriented Gradients (HOGs) de todas as imagens de ambas as bases de dados, escritos no disco
H = Hog()
H.hog_ICMC()
H.hog_ORLFaces()

# Aplicação da Principal Component Analysis (PCA) para ambas as bases de dados, também salvos em disco
pca_icmc = PrincCompAna('icmc')
pca_orl = PrincCompAna('orl')

pca_icmc.apply_pca()
pca_orl.apply_pca()

#-----------------------------------------------------------------------------------------------------------
# Treinamento e Medidas de Qualidade da Base de Dados ORLFaces20
print("ORLFACES20")
data_name = "orl"

# Inicialização das classes de cada Classificador
K = KNN(data_name)
P = Perc(data_name)

# Criação do classificador em si, assim como seu treinamento e teste
K.knn_test()
P.perc_test()

# Calculo da matriz de confusão, assim como as medidas de Acurácia do classificador e de sua Precisão para cada classe
K.confusion()
P.confusion()


# Treinamento e Medidas de Qualidade da Base de Dados de pessoas do ICMC
print("ICMC")
data_name = "icmc"

K = KNN(data_name)
P = Perc(data_name)

K.knn_test()
P.perc_test()

K.confusion()
P.confusion()


# Treinamento e Medidas de Qualidade da Base de Dados ORLFaces20 após aplicação do PCA
print("ORLFACES20 PCA")
data_name = "orlpca"

K = KNN(data_name)
P = Perc(data_name)

K.knn_test()
P.perc_test()

K.confusion()
P.confusion()

print("ICMC PCA")
data_name = "icmcpca"

K = KNN(data_name)
P = Perc(data_name)

K.knn_test()
P.perc_test()

K.confusion()
P.confusion()






