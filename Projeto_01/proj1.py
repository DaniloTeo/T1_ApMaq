from skimage.io import imread
import skimage as sk
from skimage.feature import hog
import numpy as np
from matplotlib import pyplot as plt

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
		path = "./datasets/PessoasICMC/p"
		for folder in range(1, 21):
			p = path + str(folder) + "/"

			for i in range(10):
				if i == 0:
					img_name = p + str(folder) + ".png"
				else:
					img_name = p + str(folder) + str(i) + ".png"

				img = sk.io.imread(img_name, as_gray=True)

				hog_desc, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False, block_norm = "L1")
				#print(img_name + str(hog_desc))
				self.icmc.append(hog_desc)
		np.save("./datasets/icmc.npy", np.asarray(self.icmc))

	def hog_ORLFaces(self):
		path = "./datasets/OrlFaces20/s"
		for folder in range(1, 21):
			p = path + str(folder) + "/"

			for i in range(1, 11):
				img_name = p + str(i) + ".pgm"

				img = sk.io.imread(img_name, as_gray=True)

				hog_desc, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False, block_norm = "L1")
				#print(img_name + str(hog_desc))
				self.orl_faces.append(hog_desc)
		print(np.asarray(self.orl_faces).shape)
		np.save("./datasets/orl_faces.npy", np.asarray(self.orl_faces))

# for i in range(1,21):
# 	a = Aug(i)
# 	a.aug_set()

h = Hog()
h.hog_ORLFaces()


print(np.load("./datasets/orl_faces.npy").shape)


