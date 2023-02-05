from enum import Enum
from pathlib import Path

import numpy as np
import cv2
import eos

from lib.far_model import FARModel
import lib.io as io
from lib.texture import TextureMerger, opticalFlow, warpFlow
from lib.spherical_harmonics import generateNormalmap, generateAlbedomap


FAR_DIR = Path(__file__).resolve().parents[1]
ModelType = Enum('ModelType', 'FexMM_AU FexMM_PCA FexMM_NN BFM FaceGen FLAME')
FittingMode = Enum('FittingMode', 'iterating collective')	# Iterating is always supported. Collective is only supported if expressions are loaded as array of blendshapes (currently only with FexMM_NN).


NUM_ITERATIONS = 100	# for single image fitting: should be a multiple of 5
TEXTURE_RESOLUTION = 2048
SAMPLE_NEUTRAL_MESH = False
RENDER_VIEW = False


class FARModelLoader:
	def __init__(self, morphable_model: ModelType):
		if morphable_model in {ModelType.FexMM_AU, ModelType.FexMM_PCA, ModelType.FexMM_NN}:
			self.model_dir = FAR_DIR / 'data/FexMM'
			# landmark definitions
			self.landmark_mapper = self.model_dir / 'dlib_landmark_correspondence.txt'
			self.model_contour = self.model_dir / 'fexmm_model_contours.json'

			# Load morphablemodel
			if morphable_model == ModelType.FexMM_AU:
				pca_model = FARModel(self.model_dir / 'fexmm_id-zfd-only_100.npz')
				exp_model = FARModel(self.model_dir / 'fexmm_exp-au_17.npz')
				use_expression_pca_model = True # True: use pca model, shapes can be negative. False: use blendshapes, shapes must be none negative
			elif morphable_model == ModelType.FexMM_PCA:
				pca_model = FARModel(self.model_dir / 'fexmm_id-zfd-only_100.npz')
				exp_model = FARModel(self.model_dir / 'fexmm_exp-pca_25.npz')
				use_expression_pca_model = True
			elif morphable_model == ModelType.FexMM_NN:
				pca_model = FARModel(self.model_dir / 'fexmm_id-zfd-only_100.npz')
				exp_model = FARModel(self.model_dir / 'fexmm_exp-nn_25.npz')
				use_expression_pca_model = False
			pca_model.exp_shapes = exp_model.exp_shapes
			pca_model.triangles = exp_model.triangles	# close eyes and mouth

			# finally convert to eos
			self.model, self.edge_topology = pca_model.to_eos(expression_is_pca_model=use_expression_pca_model)
		elif morphable_model == ModelType.BFM:
			self.model_dir = FAR_DIR / 'data/BFM2017_nomouth'
			# landmark definitions
			self.landmark_mapper = self.model_dir / 'ibug_to_bfm2017-1_bfm_nomouth.txt'
			self.model_contour = self.model_dir / 'bfm2017-1_bfm_nomouth_model_contours.json'

			# Load morphablemodel with expressions
			self.model = FARModel(eos.morphablemodel.load_model(str(self.model_dir / 'bfm2017-1_bfm_nomouth_199-100.bin')))
			self.model.source_file = 'bfm2017-1_bfm_nomouth_199-100.bin'
			self.model.texcoords = np.loadtxt(str(self.model_dir / 'texcoords.txt'))
			self.model = self.model.to_eos(return_edge_topology=False)
			use_expression_pca_model = True

			#  The edge topology is used to speed up computation of the occluding face contour fitting
			self.edge_topology = eos.morphablemodel.load_edge_topology(str(self.model_dir / 'bfm2017-1_bfm_nomouth_edge_topology.json'))
		elif morphable_model == ModelType.FaceGen:
			self.model_dir = FAR_DIR / 'data/FaceGen'
			# landmark definitions
			self.landmark_mapper = self.model_dir / 'dlib_landmark_correspondence.txt'
			self.model_contour = self.model_dir / 'model_contours.json'

			# Load morphablemodel
			pca_model = FARModel(self.model_dir / 'facegen_au_80-16.npz')
			use_expression_pca_model = True

			# finally convert to eos
			self.model, self.edge_topology = pca_model.to_eos(expression_is_pca_model=use_expression_pca_model)
		elif morphable_model == ModelType.FLAME:
			self.model_dir = FAR_DIR / 'data/FLAME'
			# landmark definitions
			self.landmark_mapper = self.model_dir / 'dlib_landmark_correspondence.txt'
			self.model_contour = self.model_dir / 'model_contours.json'

			# Load morphablemodel
			pca_model = FARModel(self.model_dir / 'flame_pca_300-100.npz')
			use_expression_pca_model = True

			# finally convert to eos
			self.model, self.edge_topology = pca_model.to_eos(expression_is_pca_model=use_expression_pca_model)
		else:
			raise ValueError("unsupported model type '" + morphable_model + "'!")

		# modify model (reduce shape modes and add landmark vertices)
		pca_model = FARModel(self.model)
		pca_model.reduce_shape_modes(np.s_[:80], np.s_[:25])
		pca_model.add_landmarks_vertices(self.landmark_mapper.parents[0] / 'landmarks_68.csv', self.landmark_mapper, self.model_contour)
		self.model = pca_model.to_eos(expression_is_pca_model=use_expression_pca_model, return_edge_topology=False)

		# These two are used to fit the front-facing contour to the ibug contour landmarks
		self.contour_landmarks = eos.fitting.ContourLandmarks.load(str(self.landmark_mapper))
		self.model_contour = eos.fitting.ModelContour.load(str(self.model_contour))

		# The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids
		self.landmark_mapper = eos.core.LandmarkMapper(str(self.landmark_mapper))


class FaceAvatarReconstruction:
	def __init__(self, morphable_model: ModelType, fitting_mode: FittingMode):
		self.fitting_mode = fitting_mode

		# Prepare EOS/Far
		self.eos_model = FARModelLoader(morphable_model)

		# Prepare texture merger
		self.tex_merger = TextureMerger(TEXTURE_RESOLUTION, self.eos_model.model_dir/'segment_mask.png', self.eos_model.model_dir/'front_mask.png', self.eos_model.model_dir/'reference_texture.png', 80)


	def reconstructAvatar(self, image_path, lambda_sh: float, lambda_expr: float, out_dir: Path, mesh_name : str = 'neutral', generate_texture: bool = False):
		"""
		convenience/example function to generate a complete avatar by fitting a model to a set of images, extracting textures and merging them
		"""
		out_dir = Path(out_dir)
		# load images
		images, landmarks = io.loadImageAndLandmarksDirectory(image_path)

		print(out_dir, 'images to process:', len(images))
		out_dir.mkdir(parents=True, exist_ok=True)
		mesh, identiy_coeffs, expression_coeffs, meshs, poses = self.fitShapeAndPose(images, landmarks, lambda_sh, lambda_expr)

		print('save mesh')
		io.saveMesh(out_dir / f'{mesh_name}.obj', mesh)
		if SAMPLE_NEUTRAL_MESH:
			io.saveMesh(out_dir / 'neutral_total.obj', self.eos_model.model.draw_sample(identiy_coeffs, [], []))

		print('write data')
		self.j = {}
		io.updateDict(out_dir/'data.csv', self.j, out_dir.name, mesh_name, {
			'identity_coeffs': identiy_coeffs,
			'expression_coeffs': expression_coeffs
		})

		if generate_texture:
			textures = self.extractTextures(images, meshs, poses)
			isomap = self.mergeTextures(textures)
			# image registration / optical flow
			flow = opticalFlow(isomap, self.tex_merger.ref_tex)
			isomap = warpFlow(isomap, flow)
			# reduce light effects / spherical harmonics
			# normmap = generateNormalmap(mesh, TEXTURE_RESOLUTION)
			# io.saveImage(out_dir / 'normal.exr', normmap)
			# isomap = generateAlbedomap(isomap, normmap, 5)
			io.saveImage(out_dir / f'{mesh_name}.isomap.png', isomap)

		if RENDER_VIEW:
			camera = 0 # must be < len(images)
			rendering = self.renderView(mesh, isomap, poses[camera])
			rendering = cv2.resize(rendering, (images[camera].shape[1], images[camera].shape[0]))
			io.saveImage(out_dir / 'rendering.png', rendering)


	def fitShapeAndPose(self, images: list, landmarks: list, lambda_sh: float, lambda_expr: float):
		"""
		fit a morphable model to a set of images, either iterating one at a time or to all at same time
		"""
		identity_coeffs = []; expression_coeffs = []

		n = len(images)
		if n == 0 or len(landmarks) != n:
			return None, [], [], [], []
		else:
			# Fit the model
			if self.fitting_mode == FittingMode.iterating:
				s = 0
				meshs, poses, ex_coeffs = [[]]*n, [[]]*n, []
				id_coeffs = []
				identity_coeffs = [[]]*n; expression_coeffs = [[]]*n
				for i in range(int(NUM_ITERATIONS/5)):
					for j, (lmks, img) in enumerate(zip(landmarks, images)):
						m, p, id_coeffs, ex_coeffs = eos.fitting.fit_shape_and_pose(self.eos_model.model, lmks, self.eos_model.landmark_mapper,
							img.shape[1], img.shape[0], self.eos_model.edge_topology,
							self.eos_model.contour_landmarks, self.eos_model.model_contour, 5, None, lambda_sh, None,  lambda_expr, id_coeffs, ex_coeffs)
						meshs[j] = m
						poses[j] = p
						identity_coeffs[j] = id_coeffs
						expression_coeffs[j] = ex_coeffs
						# calculate median of identity and expression coefficients so they can be used as starting value in next iteration
						s = n if i else j+1
						id_coeffs = np.median(identity_coeffs[:s], axis=0)
						ex_coeffs = np.median(expression_coeffs[:s], axis=0)
				identity_coeffs = id_coeffs.tolist()
			else:	# multi image fitting
				dim = list(zip(*map(lambda i: i.shape, images)))
				meshs, poses, identity_coeffs, expression_coeffs = eos.fitting.fit_shape_and_pose(self.eos_model.model, self.eos_model.model.get_expression_model(), landmarks, self.eos_model.landmark_mapper,
					dim[1], dim[0], self.eos_model.edge_topology, self.eos_model.contour_landmarks, self.eos_model.model_contour, NUM_ITERATIONS, None, lambda_sh, None, identity_coeffs)

			# calculate median of expression coefficients from all images and generate mesh using those
			median_expression_coeffs = np.median(expression_coeffs, axis=0).tolist()
			mesh = self.eos_model.model.draw_sample(identity_coeffs, median_expression_coeffs, [])

			return mesh, identity_coeffs, median_expression_coeffs, meshs, poses


	def extractTextures(self, images: list, meshs: list, poses: list):
		"""
		extract texture (parts) viewed from camera
		"""
		print('extract textures')
		textures = []
		for num, (m, p, i) in enumerate(zip(meshs, poses, images)):
			isomap = np.swapaxes(eos.render.extract_texture(m, p, i, True, TEXTURE_RESOLUTION), 0,1)
			# normmap = generateNormalmap(m, TEXTURE_RESOLUTION)
			# isomap = generateAlbedomap(isomap, normmap, 5)
			textures.append(isomap)
		return textures


	def mergeTextures(self, textures: list):
		"""
		merge texture parts to one unified texture map
		"""
		for tex in textures:
			self.tex_merger.add(tex)
		print('merge textures')
		tex = self.tex_merger.merge(True)
		self.tex_merger.reset()
		return tex


	def renderView(self, mesh, isomap, pose):
		"""
		render the avatar from one perspective
		"""
		isomap = cv2.cvtColor(isomap, cv2.COLOR_RGB2RGBA)
		rendering = eos.render.render(mesh, pose.get_modelview(), pose.get_projection(), 1024, 1024, isomap, True, False, False)
		return np.swapaxes(rendering, 0,1)
