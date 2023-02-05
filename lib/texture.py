import numpy as np
import cv2


def opticalFlow(img1, img2):
	gray1 = cv2.cvtColor(cv2.resize(img1, (1024, 1024)), cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(cv2.resize(img2, (1024, 1024)), cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 255, 7, 11, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
	return cv2.resize(flow * (img1.shape[0] / 1024), img1.shape[0:2])

def warpFlow(img1, flow):
	# modified version of: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
	flow = -flow
	flow[:,:,0] += np.arange(flow.shape[1])
	flow[:,:,1] += np.arange(flow.shape[0])[:,np.newaxis]
	return cv2.remap(img1, flow, None, cv2.INTER_LINEAR)

class TextureMerger:
	def __init__(self, tex_resolution, cam_mask, front_mask, ref_tex, merge_threshold=80.0): # merge threshold, in degrees, from 0 to 90.  Each triangle with a view angle smaller than the given angle will be used to merge.
		merge_threshold = (-255. / 90.) * merge_threshold + 255. # map 0° to 255, 90° to 0
		self.threshold = max(0., min(255., merge_threshold))

		NUM_CAMS_SAME_ANGLE = 2
		self.cam_mask = cv2.imread(str(cam_mask))[..., 0:3]
		s = self.cam_mask.shape
		u, self.cam_mask = np.unique(self.cam_mask.reshape(-1, 3), axis=0, return_inverse=True)
		self.cam_mask = self.cam_mask.reshape(s[0:2])
		if s[0] != tex_resolution:
			self.cam_mask = cv2.resize(self.cam_mask, (tex_resolution, tex_resolution), interpolation=cv2.INTER_NEAREST)
		self.best_cams = np.zeros((u.shape[0] - 1, NUM_CAMS_SAME_ANGLE), int)
		self.front_mask = cv2.imread(str(front_mask))[..., 0]
		if self.front_mask.shape[0] != tex_resolution:
			self.front_mask = cv2.resize(self.front_mask, (tex_resolution, tex_resolution), interpolation=cv2.INTER_NEAREST)
		self.ref_tex = cv2.imread(str(ref_tex))[..., 0:3]
		if self.ref_tex.shape[0] != tex_resolution:
			self.ref_tex = cv2.resize(self.ref_tex, (tex_resolution, tex_resolution))
		self.tex_resolution = tex_resolution
		self.reset()


	def reset(self):
		self.merged_isomap = np.zeros((self.tex_resolution, self.tex_resolution, 3), dtype=np.uint8)
		self.visibility = np.zeros((self.tex_resolution, self.tex_resolution, 1), dtype=np.uint8)
		self.best_cams = np.zeros(self.best_cams.shape, int)
		self.best_isomaps = np.zeros((*self.best_cams.shape, self.tex_resolution, self.tex_resolution, 4), np.uint8)


	def add(self, isomap):
		for i in range(self.best_cams.shape[0]):
			# rate camera
			p = isomap[self.cam_mask == i]
			r = np.sum(p[..., 3] >= self.threshold)
			# select camera if rating is good enough
			if r > self.best_cams[i, 0]:
				self.best_cams[i, 1] = self.best_cams[i, 0]
				self.best_cams[i, 0] = r
				self.best_isomaps[i, 1] = self.best_isomaps[i, 0]
				self.best_isomaps[i, 0] = isomap
			elif r > self.best_cams[i, 1]:
				self.best_cams[i, 1] = r
				self.best_isomaps[i, 1] = isomap


	def merge(self, include_nose=True):
		# best camera bin blended
		m1 = self.best_isomaps[..., 3] >= self.threshold
		avg_color = np.mean(self.best_isomaps[np.logical_and(m1, self.front_mask == 255), 0:3], axis=0)
		self.merged_isomap = np.tile(avg_color, (*self.best_isomaps.shape[2:4],1)).astype(np.uint8)
		for i, images in enumerate(self.best_isomaps):
			if not include_nose and i == 0:
				continue
			isomap = images[0, ..., 0:3]
			# self.merged_isomap[self.cam_mask == i] = isomap[self.cam_mask == i]
			alpha = images[0, ..., 3]
			# flow = opticalFlow(isomap, self.ref_tex)
			# isomap = warpFlow(isomap, flow)
			# alpha = warpFlow(alpha, flow)
			mask = np.where(self.cam_mask != i, np.uint8(0), np.uint8(255))
			mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
			mask[alpha <= 0] = 0
			center = np.nonzero(mask)
			if len(center[0]) > 0:
				center = tuple(np.flip((np.max(center, axis=1) + np.min(center, axis=1)) // 2))
				self.merged_isomap = cv2.seamlessClone(isomap, self.merged_isomap, mask, center, cv2.MIXED_CLONE)
		new_avg = np.mean(self.merged_isomap[self.front_mask == 255, 0:3], axis=0)
		self.merged_isomap = np.clip(self.merged_isomap * (avg_color / new_avg), 0, 255).astype(np.uint8)
		# self.merged_isomap = cv2.bilateralFilter(self.merged_isomap, 9, 75, 75)
		return self.merged_isomap

	def merge2(self, include_nose=True):
		# best angle per pixel blended
		m1 = self.best_isomaps[..., 3] >= self.threshold
		self.merged_isomap = np.tile(np.mean(self.best_isomaps[np.logical_and(m1, self.front_mask == 255), 0:3], axis=0), (*self.best_isomaps.shape[2:4],1)).astype(np.uint8)
		for i, images in enumerate(self.best_isomaps):
			if not include_nose and i == 0:
				continue
			isomap = images[0, ..., 0:3]
			alpha = images[0, ..., 3]
			# flow = opticalFlow(isomap, self.ref_tex)
			# isomap = warpFlow(isomap, flow)
			# alpha = warpFlow(alpha, flow)
			mask = np.where(alpha > self.visibility[..., 0], np.uint8(255), np.uint8(0)) # ignore this pixel, not visible in the extracted isomap of this current frame
			mask[alpha <= self.threshold] = 0
			cond = mask > 0
			self.visibility[cond, 0] = alpha[cond] # as soon as we've seen the pixel visible once, we set it to visible.
		for images in self.best_isomaps:
			isomap = images[0, ..., 0:3]
			alpha = images[0, ..., 3]
			mask = np.where(alpha >= self.visibility[..., 0], np.uint8(255), np.uint8(0)) # ignore this pixel, not visible in the extracted isomap of this current frame
			mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
			mask[alpha <= self.threshold] = 0
			center = np.nonzero(mask)
			if len(center[0]) > 0:
				center = tuple(np.flip((np.max(center, axis=1) + np.min(center, axis=1)) // 2))
				self.merged_isomap = cv2.seamlessClone(isomap, self.merged_isomap, mask, center, cv2.MIXED_CLONE)
		#self.merged_isomap = cv2.bilateralFilter(self.merged_isomap, 9, 75, 75)
		return self.merged_isomap

	def merge3(self):
		# best angle per pixel, no blending
		m1 = self.best_isomaps[..., 3] >= self.threshold
		self.merged_isomap = np.tile(np.mean(self.best_isomaps[np.logical_and(m1, self.front_mask == 255), 0:3], axis=0), (*self.best_isomaps.shape[2:4],1)).astype(np.uint8)
		m1 = self.best_isomaps[..., 3] == np.max(self.best_isomaps[..., 3], axis=0)
		m2 = np.any(m1, axis=(0, 1))
		bi = self.best_isomaps.copy()
		bi[np.logical_not(m1)] = 0
		self.merged_isomap[m2, 0:3] = (np.sum(bi[..., 0:3], axis=(0, 1))[m2] / np.count_nonzero(m1, axis=(0, 1))[m2, None])
		#self.merged_isomap = cv2.blur(self.merged_isomap, (self.merged_isomap.shape[0]//80, self.merged_isomap.shape[1]//80))
		return self.merged_isomap

	def merge4(self):
		# average of whole map
		m1 = self.best_isomaps[..., 3] >= self.threshold
		self.merged_isomap = np.tile(np.mean(self.best_isomaps[np.logical_and(m1, self.front_mask == 255), 0:3], axis=0), (*self.best_isomaps.shape[2:4],1)).astype(np.uint8)
		m2 = np.any(m1, axis=(0, 1))
		bi = self.best_isomaps.copy()
		bi[np.logical_not(m1)] = 0
		self.merged_isomap[m2, 0:3] = (np.sum(bi[..., 0:3], axis=(0, 1))[m2] / np.count_nonzero(m1, axis=(0, 1))[m2, None])
		#self.merged_isomap = cv2.blur(self.merged_isomap, (self.merged_isomap.shape[0]//80, self.merged_isomap.shape[1]//80))
		return self.merged_isomap
