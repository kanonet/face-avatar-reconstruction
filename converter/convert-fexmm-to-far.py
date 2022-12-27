#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np

FAR_DIR = Path(__file__).resolve().parents[1]
if not str(FAR_DIR) in sys.path:
	sys.path.append(str(FAR_DIR))
from lib.far_model import FARModel


DIR = FAR_DIR / 'data/FexMM'
BLENDER_EXECUTABLE_PATH = Path('path/to/blender.exe')	# adjust path to your blender installation!

# since fexmm_zfd_only does not contain expression shapes, we have to convert identity and expressions separately
if True:	# identity
	IMPORT_FILE = DIR / 'fexmm_zfd_only.blend'
	BLENDER_SHAPES_OBJ = 'fexmm_zfd_only'
	BLENDER_MESH_OBJ = 'fexmm_zfd_only'
	EOS_SCALE_FACTOR = 1.0	# this does affect accuracy!
	# you need to specify which shape modes to export! all available keys: obj.data.shape_keys.key_blocks.keys()
	pca_shapes = [f'{i:04}_std_pca{i}' for i in range(1, 101)]
	exp_shapes = []
else: # expressions
	IMPORT_FILE = DIR / '2020_06_17_FexMM_pure.blend'
	BLENDER_SHAPES_OBJ = 'FexMM'
	BLENDER_MESH_OBJ = 'FexMM_eyes_mouth_closed_smoothed'
	EOS_SCALE_FACTOR = 100.0	# this does affect accuracy!
	# you need to specify which shape modes to export! all available keys: obj.data.shape_keys.key_blocks.keys()
	pca_shapes = [f'neu_pca_{i}' for i in range(1, 26)]
	pca_shapes += [f'neu_asym_pca_{i}' for i in range(1, 6)]
	exp_shapes = [f'exp_pca_{i}' for i in range(1, 26)]	# pca expressions
	# exp_shapes = [f'exp_pca_nn_{i}' for i in range(1, 26)]	# non negative pca expressions
	# exp_shapes = ['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_7', 'AU_9', 'AU_10', 'AU_12', 'AU_13', 'AU_14', 'AU_15', 'AU_17', 'AU_20', 'AU_23', 'AU_25', 'AU_26']	# action unit blendshape expressions

SAVE_MODEL = True
EXPORT_FILE = DIR / 'fexmm_id-zfd-only_100_2.npz' # suffix must be .json, .csv or .npz

SAVE_OBJ = False
OBJ_FILE = DIR / 'mean_face.obj'

SAVE_EDGE_TOPOLOGY = False
EDGE_TOPOLOGY_FILE = DIR / 'edge_topology.json'


def get_model(obj, mesh, pca_shapes, exp_shapes):
	# get shape modes
	def get_shape_modes(sw_list):
		keys = []
		shapes = []
		average = obj.data.shape_keys.key_blocks['Key'].data
		for sw_name in sw_list:
			shape = obj.data.shape_keys.key_blocks[sw_name].data
			keys.append(sw_name)
			shapes.append([list(shape[v.index].co - average[v.index].co) for v in obj.data.vertices])
		return [keys, shapes]
	model = FARModel()
	model.source_file = str(IMPORT_FILE)
	model.pca_shapes = get_shape_modes(pca_shapes)
	model.exp_shapes = get_shape_modes(exp_shapes)

	# get mesh
	me = mesh.data
	uv_layer = me.uv_layers.active.data
	model.vertices = [[*v.co] for v in me.vertices]
	model.texcoords = [[0, 0]] * len(model.vertices)
	model.triangles = []
	for face in me.polygons:
		idx = []
		for loop in face.loop_indices:
			idx.append(me.loops[loop].vertex_index)
			model.texcoords[idx[-1]] = [uv_layer[loop].uv[0], 1 - uv_layer[loop].uv[1]]	# we simply assume that vertices at same position also have same UVs
		model.triangles.append(idx[0:3])
		if face.loop_total == 4:	#split quad in two triangles
			model.triangles.append([idx[0], idx[2], idx[3]])

	return model


def run_inside_blender():
	import bpy

	obj = bpy.data.objects[BLENDER_SHAPES_OBJ]
	obj.matrix_world = np.identity(4)	# Remove any trafo
	obj.scale = (EOS_SCALE_FACTOR, EOS_SCALE_FACTOR, EOS_SCALE_FACTOR)
	mesh = bpy.data.objects[BLENDER_MESH_OBJ]
	mesh.matrix_world = np.identity(4)	# Remove any trafo
	mesh.scale = (EOS_SCALE_FACTOR, EOS_SCALE_FACTOR, EOS_SCALE_FACTOR)
	bpy.ops.object.transform_apply()

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
	eos_model, edge_topology = model.to_eos(identity_model_only=True)

	if SAVE_MODEL:
		model.save(EXPORT_FILE)
		print(f'Done writing {EXPORT_FILE}!')

	if SAVE_OBJ:
		eos.core.write_obj(eos_model.get_mean(), str(OBJ_FILE))
		print(f'Done writing {OBJ_FILE}!')

	if SAVE_EDGE_TOPOLOGY:
		eos.morphablemodel.save_edge_topology(edge_topology, str(EDGE_TOPOLOGY_FILE))
		print(f'Done writing {EDGE_TOPOLOGY_FILE}!')


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
