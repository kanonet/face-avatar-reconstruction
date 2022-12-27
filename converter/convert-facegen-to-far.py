#!/usr/bin/env python3

import sys
import struct
from pathlib import Path

import numpy as np
import eos

FAR_DIR = Path(__file__).resolve().parents[1]
if not str(FAR_DIR) in sys.path:
	sys.path.append(str(FAR_DIR))
from lib.far_model import FARModel


DIR = FAR_DIR / 'data/FaceGen'

TRI_FILE = DIR / "HeadHires.tri"
EGM_FILE = DIR / "HeadHires.egm"
FG_FILE = DIR / "mean_face.fg"

SAVE_MODEL = True
EXPORT_FILE = DIR / 'facegen_au_80-98.npz' # suffix must be .json, .csv or .npz

SAVE_OBJ = True
OBJ_FILE = DIR / 'mean_face.obj'

SAVE_EDGE_TOPOLOGY = False
EDGE_TOPOLOGY_FILE = DIR / 'edge_topology.json'


# https://facegen.com/dl/sdk/doc/manual/fileformats.html


def unpack(formatStr, fileContent, offset):
	s = struct.calcsize(formatStr)
	a = struct.unpack_from(formatStr, fileContent, offset)
	if len(a) == 1:
		a = a[0]
	return a, offset + s


def split_quads(q):
	# take nfrom: https://stackoverflow.com/a/46255808
	t = np.empty((q.shape[0], 2, 3), int)
	t[:,0,:] = q[:,(0,1,2)]
	t[:,1,:] = q[:,(0,2,3)]
	t.shape = (-1, 3)

	idx_to_delete = 2*np.flatnonzero(q[:,-1] == 0)+1
	t = np.delete(t, idx_to_delete, axis=0)
	return t


def load_model(tri_file, egm_file):
	with open(tri_file, mode='rb') as file: 
		fileContent = file.read()

	filetype, offset = unpack("8s", fileContent, 0)
	(V, T, Q, LV, LS, X, ext, Md, Ms, K), offset = unpack("10I", fileContent, offset)
	_, offset = unpack("16c", fileContent, offset)

	vertices, offset = unpack("3f"*(V+K), fileContent, offset)
	vertices = np.asarray(vertices).reshape(V+K, 3)

	if T > 0:
		raise NotImplementedError

	quads, offset = unpack("4I"*Q, fileContent, offset)
	quads = np.array(quads, int).reshape(Q, 4)
	triangles = split_quads(quads)

	if LV > 0 or LS > 0:
		raise NotImplementedError

	if X == 0:
		raise NotImplementedError
	else:
		texcoords, offset = unpack("2f"*X, fileContent, offset)
		texcoords = np.asarray(texcoords).reshape(X, 2)
		texcoords[:, 1] = 1 - texcoords[:, 1]
		texindex, offset = unpack("4I"*Q, fileContent, offset)
		texindex = split_quads(np.asarray(texindex).reshape(Q, 4))

	dm = {}
	for i in range(Md):
		n, offset = unpack("I", fileContent, offset)
		label, offset = unpack(f"{n}s", fileContent, offset)
		f, offset = unpack("f", fileContent, offset)
		sh, offset = unpack("3h"*V, fileContent, offset)
		dm[label.decode()[:-1]] = (np.asarray(sh, np.float32).reshape(V, 3) * f)

	sm = {}
	k = 0
	for i in range(Ms):
		n, offset = unpack("I", fileContent, offset)
		label, offset = unpack(f"{n}s", fileContent, offset)
		L, offset = unpack("I", fileContent, offset)
		vertindex, offset = unpack("I"*L, fileContent, offset)
		if L > V:
			raise ValueError
		sm[label.decode()[:-1]] = vertindex
		k += L
	if not k == K:
		raise RuntimeError

	if not len(fileContent) == offset:
		raise RuntimeError


	with open(egm_file, mode='rb') as file: 
		fileContent = file.read()

	filetype, offset = unpack("8s", fileContent, 0)
	(V, S, A, _), offset = unpack("4L", fileContent, offset)
	_, offset = unpack("40c", fileContent, offset)

	if not V == len(vertices):
		raise RuntimeError

	pca_shapes = []
	for j in range (S):
		f, offset = unpack("f", fileContent, offset)
		sh, offset = unpack("3h"*V, fileContent, offset)
		pca_shapes.append(np.asarray(sh).reshape(V, 3) * f)

	for j in range (A):
		f, offset = unpack("f", fileContent, offset)
		sh, offset = unpack("3h"*V, fileContent, offset)
		pca_shapes.append(np.asarray(sh).reshape(V, 3) * f)
	pca_shapes = np.array(pca_shapes, np.float32)

	if not len(fileContent) == offset:
		raise RuntimeError

	# only use specific AUs
#	aus =	['AU01 Inner Brow Raiser', 'AU02 Outer Brow Raiser', 'AU04 Brow Lowerer', 'AU05 Upper Lid Raiser', 'AU06 Cheek Raise', 'AU09 Nose Wrinkler',
#			'AU10 Upper Lip Raiser', 'AU12 Lip Corner Puller', 'AU13 Sharp Lip Puller', 'AU14 Dimpler', 'AU15 Lip Corner Depressor',
#			'AU17 Chin Raiser', 'AU20 Lip Stretcher', 'AU23 Lip Tightener', 'AU25 Lips Parted', 'AU26 Jaw Drop']
#	dm = {a: dm[a] for a in aus}

	model = FARModel()
	model.source_file = [str(tri_file), str(egm_file)]
	# create uniform triangle list for vertices and texcoords
	p, i, j = np.unique(np.concatenate([vertices[triangles], texcoords[texindex]], axis=2).reshape((-1,5)), axis=0, return_index=True, return_inverse=True)
	model.triangles = j.reshape((-1,3)).tolist()
	model.vertices = p[:,:3]
	model.texcoords = p[:,3:]
	model.pca_shapes = [[""]*(S+A), pca_shapes[:,triangles].reshape((len(pca_shapes),-1,3))[:,i]]
	model.exp_shapes = [list(dm.keys()), np.asarray(list(dm.values()))[:,triangles].reshape((len(list(dm.keys())),-1,3))[:,i]]
	return model


def load_shape_coeffs(fg_file):
	with open(fg_file, mode='rb') as file: 
		fileContent = file.read()

	filetype, offset = unpack("8s", fileContent, 0)

	(GV, TV, SS, SA, TS, TA, R, T), offset = unpack("8L", fileContent, offset)

	pca_shapes, offset = unpack("h"*SS, fileContent, offset)
	sh, offset = unpack("h"*SA, fileContent, offset)
	pca_shapes = np.asarray(pca_shapes + sh, np.float32) / 1000.0

	st_sh, offset = unpack("h"*TS, fileContent, offset)
	at_sh, offset = unpack("h"*TA, fileContent, offset)

	if T == 1:
		n, offset = unpack("L", fileContent, offset)
	else:
		n = 0

	if not offset + struct.calcsize("c"*n) == len(fileContent):
		raise RuntimeError
	return pca_shapes


if __name__ == '__main__':
	model = load_model(TRI_FILE, EGM_FILE)
	eos_model, edge_topology = model.to_eos()

	if SAVE_MODEL:
		model.save(EXPORT_FILE)
		print(f'Done writing {EXPORT_FILE}!')

	if SAVE_OBJ:
		eos.core.write_obj(eos_model.get_mean(), str(OBJ_FILE))
		pca_shapes = load_shape_coeffs(FG_FILE)
		eos.core.write_obj(eos_model.draw_sample(pca_shapes, []), str(DIR / "standard_face_in_modeller.obj"))
		print(f'Done writing {OBJ_FILE} and mean_face.obj!')

	if SAVE_EDGE_TOPOLOGY:
		eos.morphablemodel.save_edge_topology(edge_topology, str(EDGE_TOPOLOGY_FILE))
		print(f'Done writing {EDGE_TOPOLOGY_FILE}!')

	print("done")
