import csv
import json
import importlib
from pathlib import Path

import numpy as np

gen_edge = importlib.import_module("lib.generate-edge-topology")


def load_landmarks(landmarksPath, contourPath):
	import toml

	d =  toml.load(str(landmarksPath))["landmark_mappings"]
	a = np.full(68, -1, int)
	for k, v in d.items():
		a[int(k) - 1] = v

	with open(contourPath, 'r') as fp:
		d = json.load(fp)["model_contour"]
	for i, v in enumerate(d["right_contour"]):
		a[int(i)] = v
	for i, v in enumerate(d["left_contour"]):
		a[int(i) + 9] = v

	return a


class FARModel:
	def __init__(self, o=None):
		if isinstance(o, dict):
			self.dict = o
		elif isinstance(o, Path) or isinstance(o, str):
			self.load(o)
		elif o is None:
			self.source_file = ''
			self.triangles = []
			self.vertices = []
			self.texcoords = []
			self.pca_shapes = [[],[]]
			self.exp_shapes = [[],[]]
			self.edge_triangles = []
			self.edge_vertices = []
		else:
			self.from_eos(o)

	@property
	def dict(self):
		to_str = lambda s: str(s.tolist()) if isinstance(s, np.ndarray) else str(s)
		return {'source_file': self.source_file,
				'triangles': to_str(self.triangles),
				'vertices': to_str(self.vertices),
				'texcoords': to_str(self.texcoords),
				'pca_shapes': {k: to_str(v) for k, v in zip(self.pca_shapes[0], self.pca_shapes[1])},
				'exp_shapes': {k: to_str(v) for k, v in zip(self.exp_shapes[0], self.exp_shapes[1])},
				'edge_triangles': to_str(self.edge_triangles),
				'edge_vertices': to_str(self.edge_vertices)}

	@dict.setter
	def dict(self, dict):
		self.source_file = dict['source_file']
		self.triangles = eval(dict['triangles'])
		self.vertices = eval(dict['vertices'])
		self.texcoords = eval(dict['texcoords'])
		self.pca_shapes = [list(dict['pca_shapes'].keys()), list(map(eval, dict['pca_shapes'].values()))]
		self.exp_shapes = [list(dict['exp_shapes'].keys()), list(map(eval, dict['exp_shapes'].values()))]
		self.edge_triangles = eval(dict['edge_triangles'])
		self.edge_vertices = eval(dict['edge_vertices'])

	def save(self, file):
		file = Path(file)
		if file.suffix == '.npz':
			np.savez_compressed(file, source_file=self.source_file, triangles=self.triangles, vertices=self.vertices, texcoords=self.texcoords,
								pca_shape_keys=self.pca_shapes[0], pca_shape_modes=self.pca_shapes[1],
								exp_shape_keys=self.exp_shapes[0], exp_shape_modes=self.exp_shapes[1],
								edge_triangles=self.edge_triangles, edge_vertices=self.edge_vertices)

		elif file.suffix == '.json':
			with open(file, 'w+') as fp:
				json.dump(self.dict, fp, indent=2)

		else:
			raise NameError('file name suffix must be .json or .npz')

	def load(self, file):
		file = Path(file)
		if file.suffix == '.npz':
			dict = np.load(file)
			self.source_file = str(dict['source_file'])
			self.triangles = dict['triangles']
			self.vertices = dict['vertices']
			self.texcoords = dict['texcoords']
			self.pca_shapes = [dict['pca_shape_keys'], dict['pca_shape_modes']]
			self.exp_shapes = [dict['exp_shape_keys'], dict['exp_shape_modes']]
			self.edge_triangles = dict['edge_triangles']
			self.edge_vertices = dict['edge_vertices']

		elif file.suffix == '.json':
			with open(file, 'r') as fp:
				self.dict = json.load(fp)

		else:
			raise NameError('file name suffix must be .json or .npz')

	def to_eos(self, expression_model=None, expression_is_pca_model=True, return_edge_topology=True, identity_model_only=False):
		import eos

		if len(self.pca_shapes[0]) == 0:
			raise ValueError('model has no identity shapes')
		if not identity_model_only:
			if expression_model is not None:
				self.exp_shapes = expression_model.exp_shapes
			if len(self.exp_shapes[0]) == 0:
				raise ValueError('model has no expression blendshapes')
		print(f'identity shapes: {len(self.pca_shapes[0])}  expression blendshapes: {len(self.exp_shapes[0])}')

		# identity shape modes
		basis = np.reshape(np.ravel(self.pca_shapes[1]), [-1, len(self.pca_shapes[0])], 'F')
		eigen = np.diag(basis.T @ basis)
		basis = basis@np.diag(1 / np.sqrt(eigen))
		pca = eos.morphablemodel.PcaModel(np.ravel(self.vertices), basis, eigen, self.triangles)

		# expression shape modes
		if identity_model_only:
			exp = []
		else:
			basis = np.reshape(np.ravel(self.exp_shapes[1]), [-1, len(self.exp_shapes[0])], 'F')
			if expression_is_pca_model:
				eigen = np.diag(basis.T @ basis)
				basis = basis@np.diag(1 / np.sqrt(eigen))
				exp = eos.morphablemodel.PcaModel(np.zeros(len(self.vertices) * 3), basis, eigen, self.triangles) # pca model of expression shapes (can be negative)
			else:
				exp = [eos.morphablemodel.Blendshape(key, basis[:, i]) for i, key in enumerate(self.exp_shapes[0])] # vector of blendshapes (none negative), needed for multi image fitting

		# Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
		morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(pca, exp,
																			color_model=eos.morphablemodel.PcaModel(),
																			vertex_definitions=None,
																			texture_coordinates=self.texcoords)
		if not return_edge_topology:
			return morphablemodel_with_expressions

		#  The edge topology is used to speed up computation of the occluding face contour fitting
		if self.edge_triangles == [] or self.edge_vertices == []:
			self.edge_triangles, self.edge_vertices = gen_edge.generate_edge_topology(self.triangles)
		edge_topology = eos.morphablemodel.EdgeTopology(self.edge_triangles, self.edge_vertices)

		return morphablemodel_with_expressions, edge_topology

	def from_eos(self, model):
		p = model.get_shape_model()
		p = np.reshape(p.get_rescaled_pca_basis(), (3, -1, p.get_num_principal_components()), 'F').T
		self.pca_shapes = [np.arange(len(p)), p]
		p = model.get_expression_model()
		if str(model.get_expression_model_type()) == "ExpressionModelType.PcaModel":
			p = np.reshape(p.get_rescaled_pca_basis(), (3, -1, p.get_num_principal_components()), 'F').T
			self.exp_shapes = [np.arange(len(p)), p]
		else:
			self.exp_shapes = [[b.name for b in p], np.reshape([b.deformation for b in p], (len(p), -1, 3))]
		m = model.get_mean()
		self.vertices = m.vertices
		self.triangles = m.tvi
		self.texcoords = m.texcoords
		self.edge_triangles, self.edge_vertices = [], []

	def reduce_shape_modes(self, num_id_shapes_asym, num_exp_shapes, num_id_shapes_sym=None):
		# usually shape modes are ordered by importance, so we can just cut away some of the later ones without losing much details
		# but for some versions of FexMM ID shapes are not completly ordered by importance, since symetric and asymetric shapes are grouped
		# in that case we need to cut modes from those groups seperately
		if not (isinstance(num_id_shapes_asym, slice) and isinstance(num_exp_shapes, slice) and (isinstance(num_id_shapes_sym, slice) or num_id_shapes_sym is None)):
			raise TypeError("input must be instance of numpy.s_[]")
		if num_id_shapes_sym is None:
			self.pca_shapes[0] = self.pca_shapes[0][num_id_shapes_asym]	# sym id shapes only
			self.pca_shapes[1] = self.pca_shapes[1][num_id_shapes_asym]
		else:
			self.pca_shapes[0] = np.concatenate([self.pca_shapes[0][num_id_shapes_asym], self.pca_shapes[0][num_id_shapes_sym]])	# sym + asym id shapes
			self.pca_shapes[1] = np.concatenate([self.pca_shapes[1][num_id_shapes_asym], self.pca_shapes[1][num_id_shapes_sym]])
		self.exp_shapes[0] = self.exp_shapes[0][num_exp_shapes]	# expression shapes
		self.exp_shapes[1] = self.exp_shapes[1][num_exp_shapes]

	def add_landmarks_vertices(self, pathBarycentricLandmarks, pathIbugLandmarks, pathContours):
		lms = load_landmarks(pathIbugLandmarks, pathContours)
		with open(pathBarycentricLandmarks) as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for lm, x, y, z, triangle, b1, b2, b3 in reader:
				# generate new vertex at barycentric landmark position and place it at the end of lists for vertices, texcoords and each shape mode
				b1, b2, b3 = float(b1), float(b2), float(b3)
				t = self.triangles[int(triangle)]
				i = len(self.vertices)
				t1, t2, t3 = np.asarray(self.vertices)[t]
				self.vertices = np.concatenate([self.vertices, [t1*b1 + t2*b2 + t3*b3]], axis=0)
				if len(self.texcoords) > 0:
					t1, t2, t3 = np.asarray(self.texcoords)[t]
					self.texcoords = np.concatenate([self.texcoords, [t1*b1 + t2*b2 + t3*b3]], axis=0)
				t1, t2, t3 = np.hsplit(self.pca_shapes[1][:, t, :], 3)
				self.pca_shapes[1] = np.concatenate([self.pca_shapes[1], t1*b1 + t2*b2 + t3*b3], axis=1)
				t1, t2, t3 = np.hsplit(self.exp_shapes[1][:, t, :], 3)
				self.exp_shapes[1] = np.concatenate([self.exp_shapes[1], t1*b1 + t2*b2 + t3*b3], axis=1)

				# split triangle in 3 (not needed, since vertices dont need to be part of a triangle for eos to function)
				# t1, t2, t3 = t
				# self.triangles = np.concatenate([self.triangles, [[t1, t2, i]]], axis=0)
				# self.triangles = np.concatenate([self.triangles, [[t2, t3, i]]], axis=0)
				# self.triangles[int(triangle)] = [t3, t1, i]

				# inside vertices, texcoods and shape modes array, swap new vertex index with old one of that landmark
				# this way landmark mapper and modell contour do not need to get changed
				lm = lms[int(lm)]
				self.triangles = np.asarray(self.triangles)
				g = self.triangles == lm
				h = self.triangles == i
				self.triangles[h] = lm
				self.triangles[g] = i
				self.vertices[lm], self.vertices[i] = self.vertices[i], self.vertices[lm].copy()
				if len(self.texcoords) > 0:
					self.texcoords[lm], self.texcoords[i] = self.texcoords[i], self.texcoords[lm].copy()
				self.pca_shapes[1][:,lm], self.pca_shapes[1][:,i] = self.pca_shapes[1][:,i], self.pca_shapes[1][:,lm].copy()
				self.exp_shapes[1][:,lm], self.exp_shapes[1][:,i] = self.exp_shapes[1][:,i], self.exp_shapes[1][:,lm].copy()
		self.edge_triangles, self.edge_vertices = [], []
