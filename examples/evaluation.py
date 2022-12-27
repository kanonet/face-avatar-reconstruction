import csv, json
from math import sqrt
from pathlib import Path

import numpy as np
import cv2
import cairo
import trimesh

import eos
from lib.far_fitting import ModelType


MAX_VALUE = 10.0
FAR_DIR = Path(__file__).resolve().parents[1]
MASK_BORDER_VERTICES = True	# cut away vertices outside of mask = do not calculate RMSE of whole mesh
RENDER_DISTANCE_MAP = False
USE_MANUAL_ANTHROPOMETRIC_LANDMARKS = False
SAVE_REGISTERED_MESHS = False


def loadObj(path):
	mesh = eos.core.read_obj(str(path))
	if np.any(mesh.tvi != mesh.tti):
		assert(len(mesh.tvi) == len(mesh.tti))
		texcoords = np.zeros((len(mesh.tvi), 2))
		texcoords[mesh.tvi] = np.array(mesh.texcoords)[mesh.tti]
	else:
		texcoords = mesh.texcoords
	return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.tvi, visual=trimesh.visual.TextureVisuals(uv=texcoords), process=False)

def loadLandmarks(landmarksPath, contourPath):
	import toml

	d =  toml.load(str(landmarksPath))['landmark_mappings']
	a = np.full(68, -1, int)
	for k, v in d.items():
		a[int(k) - 1] = v

	with open(contourPath, 'r') as fp:
		d = json.load(fp)['model_contour']
	for i, v in enumerate(d['right_contour']):
		a[int(i)] = v
	for i, v in enumerate(d['left_contour']):
		a[int(i) + 9] = v

	return a

def loadVirtualLandmarks(pathBaryzentricLandmarks):
	a = np.full((68, 4), -1, float)
	with open(pathBaryzentricLandmarks) as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for i, (lm, x, y, z, triangle, b1, b2, b3) in enumerate(reader):
			a[i] = [int(triangle), float(b1), float(b2), float(b3)]
	return a

def extractLandmarks3d(mesh, index):
	ret = np.full((68, 3), np.nan, float)
	if index.ndim == 1:
		i = index > -1
		ret[i] = mesh.vertices[index[i]]
	else:
		i = index[:, 0] > -1
		ret[i] = np.sum(np.multiply(mesh.triangles[index[i, 0].astype(int)], index[i, 1:, None]), axis=1)
	return ret

def loadAnthropometricLandmarks(path):
	d = {}
	with open(path) as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for lm, x, y, z, triangle, bx, by, bz in reader:
			d[lm] = [int(triangle), [float(bx), float(by), float(bz)]]
	return d

def savePickedPoints(path, lm_names, lm_coords):
	with open(path, 'w') as pp_file:
		pp_file.write('<!DOCTYPE PickedPoints>\n')
		pp_file.write('<PickedPoints>\n')
		for lm, (x, y, z) in zip(lm_names, lm_coords):
			pp_file.write(f'	<point x="{x}" y="{y}" z="{z}" name="{lm}"/>\n')
		pp_file.write('</PickedPoints>\n')

class TestBed:
	def __init__(self, faceModel: ModelType):
		self.faceModel = faceModel
		self.anthropometric_landmarks = None

		# load landmarks
		if self.faceModel == ModelType.FaceGen:
			# self.landmarks_fitted = loadLandmarks(FAR_DIR / 'data/FaceGen/dlib_landmark_correspondence.txt', FAR_DIR / 'data/FaceGen/model_contours.json')
			self.landmarks_fitted = loadVirtualLandmarks(FAR_DIR / 'data/FaceGen/landmarks_68.csv')
			self.anthropometric_landmarks = loadAnthropometricLandmarks(FAR_DIR / 'data/FaceGen/landmarks_anthropometric.csv')
		elif self.faceModel == ModelType.FLAME:
			# self.landmarks_fitted = loadLandmarks(FAR_DIR / 'data/FLAME/dlib_landmark_correspondence.txt', FAR_DIR / 'data/FLAME/model_contours.json')
			self.landmarks_fitted = loadVirtualLandmarks(FAR_DIR / 'data/FLAME/landmarks_68.csv')
			self.anthropometric_landmarks = loadAnthropometricLandmarks(FAR_DIR / 'data/FLAME/landmarks_anthropometric.csv')
		elif self.faceModel == ModelType.BFM:
			# self.landmarks_fitted = loadLandmarks(FAR_DIR / 'data/BFM2017_nomouth/ibug_to_bfm2017-1_bfm_nomouth.txt', FAR_DIR / 'data/BFM2017_nomouth/bfm2017-1_bfm_nomouth_model_contours.json')
			self.landmarks_fitted = loadVirtualLandmarks(FAR_DIR / 'data/BFM2017_nomouth/landmarks_68.csv')
			self.anthropometric_landmarks = loadAnthropometricLandmarks(FAR_DIR / 'data/BFM2017_nomouth/landmarks_anthropometric.csv')
		elif self.faceModel in {ModelType.FexMM_AU, ModelType.FexMM_PCA, ModelType.FexMM_NN}:
			# self.landmarks_fitted = loadLandmarks(FAR_DIR / 'data/FexMM/dlib_landmark_correspondence.txt', FAR_DIR / 'data/FexMM/fexmm_model_contours.json')
			self.landmarks_fitted = loadVirtualLandmarks(FAR_DIR / 'data/FexMM/landmarks_68.csv')
			self.anthropometric_landmarks = loadAnthropometricLandmarks(FAR_DIR / 'data/FexMM/landmarks_anthropometric.csv')
		else:
			raise Exception("unsupported model type '" + self.faceModel + "'!")

		self.landmarks_selected = [36,39,42,45,31,35,30,48,54,51,57]	#,18,21,22,25	# 0 based index!
		# self.landmarks_reference68 = loadLandmarks(FACE_VR_DIR / 'data/FexMM/dlib_landmark_correspondence.txt', FACE_VR_DIR / 'data/FexMM/fexmm_model_contours.json')
		self.landmarks_reference68 = loadVirtualLandmarks(FAR_DIR / 'data/FexMM/landmarks_68.csv')
		self.landmarks_ref_anth = loadAnthropometricLandmarks(FAR_DIR / 'data/FexMM/landmarks_anthropometric.csv')

		# load mesh mask image
		self.maskImage = np.flip(cv2.imread(str(FAR_DIR/'data/FexMM/face_mask.png'))[..., 0], [0,1])
		
		# load mean faceID
		self.mesh_mean = loadObj(FAR_DIR / 'data/FexMM/mean_face.obj')
		self.mesh_mean.landmarks = extractLandmarks3d(self.mesh_mean, self.landmarks_reference68)


	def loadMeshs(self, mesh_name, fittedDir, referenceDir):
		# load reference mesh (reference_matched is used to get the transformation to mean face, reference_full is used to calculate distance)
		mesh_reference_matched = loadObj(Path(referenceDir) / 'mesh.obj')
		mesh_reference_full = loadObj(Path(referenceDir) / 'mesh_matched.obj')

		# load fitted mesh
		if self.faceModel in {ModelType.BFM, ModelType.FaceGen, ModelType.FexMM_AU, ModelType.FexMM_PCA, ModelType.FexMM_NN, ModelType.FLAME}:
			mesh_fitted = loadObj(Path(fittedDir) / f'{mesh_name}.obj')
		else:
			raise Exception("unsupported model type '" + self.faceModel + "'!")

		# extract landmarks
		mesh_fitted.landmarks = extractLandmarks3d(mesh_fitted, self.landmarks_fitted)
		mesh_reference_matched.landmarks = extractLandmarks3d(mesh_reference_matched, self.landmarks_reference68)
		anth_lm = list(zip(*self.anthropometric_landmarks.values()))
		anth_lm = trimesh.triangles.barycentric_to_points(mesh_fitted.triangles[np.array(anth_lm[0])], np.array(anth_lm[1]))
		mesh_fitted.landmarks = np.concatenate([mesh_fitted.landmarks, anth_lm])
		if USE_MANUAL_ANTHROPOMETRIC_LANDMARKS:
			with open(Path(referenceDir) / 'gt.csv', newline='') as csvfile:
				mesh_reference_full.landmarks = np.concatenate([mesh_reference_matched.landmarks, np.full((len(self.anthropometric_landmarks), 3), np.nan, float)])
				reader = csv.reader(csvfile, delimiter=',')
				for _, lm, x, y, z in reader:
					if lm in self.anthropometric_landmarks:
						mesh_reference_full.landmarks[list(self.anthropometric_landmarks.keys()).index(lm) + 68] = [float(x), float(y), float(z)]
		else:
			anth_lm = list(zip(*self.landmarks_ref_anth.values()))
			anth_lm = trimesh.triangles.barycentric_to_points(mesh_reference_matched.triangles[np.array(anth_lm[0])], np.array(anth_lm[1]))
			mesh_reference_full.landmarks = np.concatenate([mesh_reference_matched.landmarks, anth_lm])

		return mesh_reference_matched, mesh_reference_full, mesh_fitted


	def maskMesh(self, mesh):
		# return mask for list of vertices of mesh
		t_y, t_x = np.squeeze(np.dsplit(np.clip((mesh.visual.uv[mesh.faces] * self.maskImage.shape[:2]).round().astype(int), 0, self.maskImage.shape[1]-1), 2))
		mask = self.maskImage[t_x, t_y] > 50
		mesh.update_faces(np.any(mask, axis=1))
		mesh.remove_unreferenced_vertices()
		return mesh


	def transformMesh(self, mesh, mat):
		# apply transformation
		mesh.apply_transform(mat)
		mesh.landmarks = trimesh.transformations.transform_points(mesh.landmarks, mat)
		return mesh


	def getDistance(self, mesh_reference, mesh_fitted):
		# calculate distance
		dm = trimesh.proximity.closest_point(mesh_fitted, mesh_reference.vertices)[1]

		# calculate mesh rmse
		rmse_m = np.linalg.norm(dm) / sqrt(len(dm))

		# calculate landmark rmse
		dl = np.linalg.norm(mesh_reference.landmarks - mesh_fitted.landmarks, axis=1)
		rmse_l = np.linalg.norm(dl[self.landmarks_selected]) / sqrt(len(self.landmarks_selected))

		lm_names = [f'lm_{i}' for i in range(1, 69)]
		if self.anthropometric_landmarks:
			lm_names += list(self.anthropometric_landmarks.keys())
		dl = dict(zip(lm_names, dl))
		return dl, dm, rmse_l, rmse_m


	def renderDistance(self, mesh, d, texture_size=1024):
		assert len(d) == len(mesh.vertices)
		faces = np.asarray(mesh.faces)
		tex = np.asarray(mesh.visual.uv)[faces] * texture_size
		# calculate normal per vertex
		norm = np.zeros((len(mesh.vertices), 3), np.float32)
		norm[...,0] = 1.0
		norm[..., 1] = np.clip(np.abs(d)/MAX_VALUE, 0.0, 1.0)
		norm[..., 2] = norm[..., 1]
		surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, texture_size, texture_size)
		pattern = cairo.MeshPattern()
		context = cairo.Context(surface)
		for t, n in zip(tex, norm[faces]):
			pattern.begin_patch()
			pattern.move_to(*t[0])
			pattern.line_to(*t[1])
			pattern.line_to(*t[2])
			pattern.set_corner_color_rgb(0, *n[0])
			pattern.set_corner_color_rgb(1, *n[1])
			pattern.set_corner_color_rgb(2, *n[2])
			pattern.end_patch()
		context.set_source(pattern)
		context.rectangle(0,0,texture_size,texture_size)
		context.fill()
		surface.flush()
		# convert cairo surface to numpy/OpenCV
		normmap = np.ndarray((texture_size,texture_size,4), np.uint8, buffer=surface.get_data()).astype(np.float32)
		cond = normmap[..., 2] > 0.0
		normmap[cond, 0:2] = 255 - normmap[cond, 0:2]
		return np.flip(normmap, [0,1])


	def evaluate(self, mesh_name, fitted_dir, reference_dir):
		fitted_dir = Path(fitted_dir)
		fitted_dir.mkdir(parents=True, exist_ok=True)

		# load meshs
		mesh_reference_matched, mesh_reference_full, mesh_fitted = self.loadMeshs(mesh_name, fitted_dir, reference_dir)

		# mask border vertices of both reference meshs
		if MASK_BORDER_VERTICES:
			mesh_reference_full = self.maskMesh(mesh_reference_full)
			mesh_reference_matched = self.maskMesh(mesh_reference_matched)

		# transform references face to mean face -> same scale for all references
		mat, _, _ = trimesh.registration.procrustes(mesh_reference_matched.landmarks[self.landmarks_selected], self.mesh_mean.landmarks[self.landmarks_selected])
		self.transformMesh(mesh_reference_matched, mat)
		self.transformMesh(mesh_reference_full, mat)

		# transform fitted face to reference
		mat, _, _ = trimesh.registration.procrustes(mesh_fitted.landmarks[self.landmarks_selected], mesh_reference_matched.landmarks[self.landmarks_selected])
		self.transformMesh(mesh_fitted, mat)
		mat, _, _ = trimesh.registration.icp(mesh_reference_full.vertices, mesh_fitted, threshold=5e-04, max_iterations=150)
		self.transformMesh(mesh_fitted, np.linalg.inv(mat))

		# save meshes for debugging
		if SAVE_REGISTERED_MESHS:
			with open(str(fitted_dir / 'fit_reg.obj'), 'w', encoding='utf-8') as f1, open(str(fitted_dir / 'ref_reg.obj'), 'w', encoding='utf-8') as f2:
				mesh_fitted.export(f1, file_type='obj', include_texture=False)
				mesh_reference_full.export(f2, file_type='obj', include_texture=False)
			with open(str(fitted_dir / 'match_reg.obj'), 'w', encoding='utf-8') as f3:
				mesh_reference_matched.export(f3, file_type='obj', include_texture=False)
			lm_names = [f'lm_{i}' for i in range(1, 69)]
			if self.anthropometric_landmarks:
				lm_names += list(self.anthropometric_landmarks.keys())
			savePickedPoints(fitted_dir / 'landmarks_ref_reg.pp', lm_names, mesh_reference_full.landmarks)
			savePickedPoints(fitted_dir / 'landmarks_fit_reg.pp', lm_names, mesh_fitted.landmarks)

		# calculate distance and rmse
		dl, dm, rmse_l, rmse_m = self.getDistance(mesh_reference_full, mesh_fitted)

		# render distance map
		if RENDER_DISTANCE_MAP:
			img = self.renderDistance(mesh_reference_full, dm)
			cv2.imwrite(str(fitted_dir / f'diff-{mesh_name}.png'), img)

		return rmse_l, rmse_m, dm.min(), dm.max(), dl
