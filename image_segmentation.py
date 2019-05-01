import numpy as np
from matplotlib import pyplot as plt
import cv2

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN

from sklearn.mixture import GaussianMixture


class Clustering():
	def __init__(self, img, image):
		self.img = img
		self.image = image

	def K_Means(self):
		km = KMeans(n_clusters=4,max_iter=300)
		km.fit(self.image)

		centers = km.cluster_centers_

		point_distances = cdist(centers, self.image, 'euclidean')
		cluster_indexes = np.argmin(point_distances, axis=0)
		segmented = centers[cluster_indexes]

		segmented_image = segmented.reshape(self.img.shape).astype(np.uint8)

		return segmented_image

	def Mean_Shift(self):
		bandwidth = estimate_bandwidth(self.image, quantile=0.1, n_samples=400)
		print( bandwidth)
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=50)
		ms.fit(self.image)

		number_of_clusters = len(np.unique(ms.labels_))
		print("Number of Clusters : ", number_of_clusters)

		centers = ms.cluster_centers_

		point_distances = cdist(centers, self.image, 'euclidean')
		cluster_indexes = np.argmin(point_distances, axis=0)
		segmented = centers[cluster_indexes]

		segmented_image = segmented.reshape(self.img.shape).astype(np.uint8)
		
		return segmented_image


	def Agglomerative_Clustering(self):
		connectivity = grid_to_graph(*img.shape[:2])

		ward = AgglomerativeClustering(n_clusters=4,linkage='ward',connectivity=connectivity)
		ward.fit(self.image)
		number_of_clusters = len(np.unique(ward.labels_))
		print("Number of clusters : ", number_of_clusters)

		centers = np.zeros((number_of_clusters, 3))
		for i in range(0, number_of_clusters):
		    cluster_points = self.image[ward.labels_ == i]
		    cluster_mean = np.mean(cluster_points, axis=0)
		    centers[i, :] = cluster_mean
		labels = ward.labels_
		# print(centers)

		segmented = centers[labels]

		segmented_image = segmented.reshape(self.img.shape).astype(np.uint8)
		return segmented_image


	def DB_SCAN(self):
		db = DBSCAN(eps=255*0.01, min_samples=20, metric='euclidean')
		db.fit(self.image)

		number_of_clusters = np.max(db.labels_) + 1
		print("Number of clusters : ", number_of_clusters)

		centers = np.zeros((number_of_clusters, 3))
		for i in range(0, number_of_clusters):
		    cluster_points = self.image[db.labels_ == i]
		    cluster_mean = np.mean(cluster_points, axis=0)
		    centers[i, :] = cluster_mean

		point_distances = cdist(centers, self.image, 'euclidean')
		cluster_indexes = np.argmin(point_distances, axis=0)

		segmented = centers[cluster_indexes]
	
		segmented_image = segmented.reshape(self.img.shape).astype(np.uint8)
		return segmented_image

	def GMM(self):
		gmm = GaussianMixture(covariance_type='full', n_components=5)
		gmm.fit(self.image)

		labels = gmm.predict(self.image)

		number_of_clusters = len(np.unique(labels))
		print("Number of clusters : ", number_of_clusters)


		centers = np.zeros((number_of_clusters, 3))
		for i in range(0, number_of_clusters):
		    cluster_points = self.image[labels == i]
		    cluster_mean = np.mean(cluster_points, axis=0)
		    centers[i, :] = cluster_mean

		point_distances = cdist(centers, self.image, 'euclidean')
		cluster_indexes = np.argmin(point_distances, axis=0)

		segmented = centers[cluster_indexes]
	
		segmented_image = segmented.reshape(self.img.shape).astype(np.uint8)
		return segmented_image


if __name__ == "__main__":
	img = cv2.imread("../image/61060.jpg",1)
	print("Image shape : ",img.shape)

	image = img.reshape(-1, img.shape[-1])
	print("Image shape after flatening", image.shape)

	cluster = Clustering(img, image)

	print("KMEANS CLUSTERING")
	segmented_image = cluster.K_Means()
	cv2.imwrite('KMeans.jpg',segmented_image)

	print("MEANSHIFT CLUSTERING")
	segmented_image = cluster.Mean_Shift()
	cv2.imwrite('MeanShift.jpg',segmented_image)

	print("AGGLOMERATIVE CLUSTERING")
	segmented_image = cluster.Agglomerative_Clustering()
	cv2.imwrite('Agglom.jpg',segmented_image)

	print("DBSCAN CLUSTERING")
	segmented_image = cluster.DB_SCAN()
	cv2.imwrite('DBSCAN.jpg',segmented_image)

	print("GAUSSIAN MIXTURE MODEL")
	segmented_image = cluster.GMM()
	cv2.imwrite('GMM.jpg',segmented_image)