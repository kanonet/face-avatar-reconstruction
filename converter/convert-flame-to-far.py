#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np

FAR_DIR = Path(__file__).resolve().parents[1]
if not str(FAR_DIR) in sys.path:
	sys.path.append(str(FAR_DIR))
from lib.far_model import FARModel


DIR = FAR_DIR / 'data/FLAME'
BLENDER_EXECUTABLE_PATH = Path('path/to/blender.exe')	# adjust path to your blender installation!

IMPORT_FILE = DIR / 'flame_model.blend'
BLENDER_SHAPES_OBJ = 'FLAME-generic'
BLENDER_MESH_OBJ = 'FLAME-generic'
EOS_SCALE_FACTOR = 1000.0	# this does affect accuracy!
TEXTURE_FILE = DIR / 'texture_data_256.npy'
LANDMARK_FILE = DIR / 'flame_static_embedding_68.pkl'

SAVE_MODEL = True
EXPORT_FILE = DIR / 'flame_pca_300-100.npz' # suffix must be .json, .csv or .npz

SAVE_OBJ = True
OBJ_FILE = DIR / 'mean_face.obj'

SAVE_EDGE_TOPOLOGY = False
EDGE_TOPOLOGY_FILE = DIR / 'edge_topology.json'

SAVE_LANDMARKS = True


def get_model(obj, mesh, pca_shapes, exp_shapes):
	model = FARModel()
	model.source_file = [str(IMPORT_FILE)]

	# get shape modes
	def get_shape_modes(sw_list):
		keys = []
		shapes = []
		average = obj.data.shape_keys.key_blocks['Base'].data
		for sw_name in sw_list:
			shape = obj.data.shape_keys.key_blocks[sw_name].data
			keys.append(sw_name)
			shapes.append([list(shape[v.index].co - average[v.index].co) for v in obj.data.vertices])
		return [keys, np.array(shapes)]
	model.pca_shapes = get_shape_modes(pca_shapes)
	model.exp_shapes = get_shape_modes(exp_shapes)

	# get mesh
	me = mesh.data
	vertices = np.array([[*v.co] for v in me.vertices])
	triangles = []
	for face in me.polygons:
		idx = []
		for loop in face.loop_indices:
			idx.append(me.loops[loop].vertex_index)
		triangles.append(idx[0:3])
		if face.loop_total == 4:	#split quad in two triangles
			triangles.append([idx[0], idx[2], idx[3]])

	# load uv coords
	texture_data = np.load(TEXTURE_FILE, allow_pickle=True, encoding='latin1').item()
	texcoords = texture_data['vt']
	texcoords[:, 1] = 1 - texcoords[:, 1]
	texindex = texture_data['ft']

	# create uniform triangle list for vertices and texcoords
	p, i, j = np.unique(np.concatenate([vertices[triangles], texcoords[texindex]], axis=2).reshape((-1,5)), axis=0, return_index=True, return_inverse=True)
	model.triangles = j.reshape((-1,3)).tolist()
	model.vertices = p[:,:3]
	model.texcoords = p[:,3:]
	model.pca_shapes[1] = model.pca_shapes[1][:,triangles].reshape((len(model.pca_shapes[1]),-1,3))[:,i]
	model.exp_shapes[1] = model.exp_shapes[1][:,triangles].reshape((len(model.exp_shapes[1]),-1,3))[:,i]

	return model


def convert_landmarks(mesh):
	import pickle, csv
	import trimesh

	with open(LANDMARK_FILE, 'rb') as f:
		dd = pickle.load(f, encoding='latin1')
	mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.tvi, visual=None, process=False)
	points = trimesh.triangles.barycentric_to_points(mesh.triangles[dd['lmk_face_idx']], dd['lmk_b_coords'])

	# save csv
	with open(DIR / 'landmarks_68.csv', 'w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
		for lm, ((x, y, z), t, (bx, by, bz)) in enumerate(zip(points, dd['lmk_face_idx'], dd['lmk_b_coords'])):
			writer.writerow([lm, x, y, z, t, bx, by, bz])

	# save pp
	with open(DIR / 'landmarks_68.pp', 'w') as pp_file:
		pp_file.write('<!DOCTYPE PickedPoints>\n')
		pp_file.write('<PickedPoints>\n')
		for lm, (x, y, z) in enumerate(points):
			pp_file.write(f'	<point x="{x}" y="{y}" z="{z}" name="{lm}"/>\n')
		pp_file.write('</PickedPoints>\n')

	# find vertices closest to landmarks
	d, idx = trimesh.proximity.ProximityQuery(mesh).vertex(points)
	vert = mesh.vertices[idx]

	# ensure landmark mirroring
	lm_correspondence = [
		[0,16],[1,15],[2,14],[3,13],[4,12],[5,11],[6,10],[7,9],
		[17,26],[18,25],[19,24],[20,23],[21,22],
		[31,35],[32,34],
		[36,45],[37,44],[38,43],[39,42],[40,47],[41,46],
		[48,54],[49,53],[50,52],[55,59],[56,58],[60,64],[61,63],[65,67]
	]
	mirror_axis = [-1, 1, 1]
	for c1, c2 in lm_correspondence:
		if not np.allclose(vert[c1] * mirror_axis, vert[c2]):
			if d[c1] < d[c2]:
				vert[c2] = vert[c1] * mirror_axis
			else:
				vert[c1] = vert[c2] * mirror_axis
	_, idx = trimesh.proximity.ProximityQuery(mesh).vertex(vert)
	vert = mesh.vertices[idx]

	# save csv
	with open(DIR / 'landmarks.csv', 'w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
		for (x, y, z), i in zip(vert, idx):
			writer.writerow([x, y, z, i])

	# save pp
	with open(DIR / 'landmarks.pp', 'w') as pp_file:
		pp_file.write('<!DOCTYPE PickedPoints>\n')
		pp_file.write('<PickedPoints>\n')
		for i, (x, y, z) in enumerate(vert):
			pp_file.write(f'	<point x="{x}" y="{y}" z="{z}" name="{i+1}"/>\n')
		pp_file.write('</PickedPoints>\n')


def run_inside_blender():
	import bpy

	obj = bpy.data.objects[BLENDER_SHAPES_OBJ]
	obj.matrix_world = np.identity(4)	# Remove any trafo
	obj.rotation_euler = (np.radians(-90), 0, 0)
	obj.scale = (EOS_SCALE_FACTOR, EOS_SCALE_FACTOR, EOS_SCALE_FACTOR)
	mesh = bpy.data.objects[BLENDER_MESH_OBJ]
	mesh.matrix_world = np.identity(4)	# Remove any trafo
	mesh.rotation_euler = (np.radians(-90), 0, 0)
	mesh.scale = (EOS_SCALE_FACTOR, EOS_SCALE_FACTOR, EOS_SCALE_FACTOR)
	bpy.ops.object.transform_apply()

	# all available keys: obj.data.shape_keys.key_blocks.keys()
	pca_shapes = [f'Shape{i}' for i in range(1, 301)]
	exp_shapes = [f'Exp{i}' for i in range(1, 101)]

	model = get_model(obj, mesh, pca_shapes, exp_shapes)

	print('serialized model:', json.dumps(model.dict), flush=True)


def run_after_blender(out):
	# process blender output
	for s in out.splitlines():
		if s[0:17] == b'serialized model:':
			model = FARModel(json.loads(s[18:]))
		else:
			print(s.decode())
	print('')
	import eos
	eos_model, edge_topology = model.to_eos()

	if SAVE_MODEL:
		model.save(EXPORT_FILE)
		print(f'Done writing {EXPORT_FILE}!')

	if SAVE_OBJ:
		eos.core.write_obj(eos_model.get_mean(), str(OBJ_FILE))
		print(f'Done writing {OBJ_FILE}!')

	if SAVE_EDGE_TOPOLOGY:
		eos.morphablemodel.save_edge_topology(edge_topology, str(EDGE_TOPOLOGY_FILE))
		print(f'Done writing {EDGE_TOPOLOGY_FILE}!')

	if SAVE_LANDMARKS:
		convert_landmarks(eos_model.get_mean())


if __name__ == '__main__':
	if len(sys.argv) == 1:  # script was not called in blender
		# start blender and run script
		from subprocess import check_output
		out = check_output([str(BLENDER_EXECUTABLE_PATH), str(IMPORT_FILE), '-b', '-P', __file__])
		run_after_blender(out)

	elif len(sys.argv) > 1:	# this script runs inside blender (most likely)
		run_inside_blender()

	else:
		raise RuntimeError('This should never happen!')
