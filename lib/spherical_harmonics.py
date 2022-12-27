import numpy as np
import cv2


def generateSphericalHarmonics1(n):
	"""
	generate the first 9 spherical harmonic coefficients from normals
	normals must be normalized and in range [-1,1], shape must be (..., 3)
	"""
	# source: https://github.com/royshil/HeadReplacement/blob/master/HeadReplacement/spherical_harmonics_analysis.h
	nx, ny, nz = n[...,0], n[...,1], n[...,2]
	nxs = nx * nx
	nys = ny * ny
	return np.array([
		[0.28209479177387814] * len(nx), # 1.0 / np.sqrt(4 * np.pi)
		1.0233267079464883 * nz, # ((2 * np.pi) / 3) * np.sqrt(3 / (4 * np.pi))
		1.0233267079464883 * ny,
		1.0233267079464883 * nx,
		0.24770795610037571 * (2.0*nz*nz - nxs - nys), # (np.pi / 4) * (1 / 2) * np.sqrt(5 / (4 * np.pi))
		0.8580855308097834 * ny * nz, # (np.pi / 4) * 3 * np.sqrt(5 / (12 * np.pi))
		0.8580855308097834 * nx * nz,
		0.8580855308097834 * nx * ny,
		0.4290427654048917 * (nxs - nys)	# (np.pi / 4) * (3 / 2) * np.sqrt(5 / (12 * np.pi))
	], np.float32)

def generateSphericalHarmonics2(n):
	"""
	generate the first 9 spherical harmonic coefficients from normals
	normals must be normalized and in range [-1,1], shape must be (..., 3)
	"""
	# source : https://pdfs.semanticscholar.org/83d9/28031e78f15d9813061b53d25a4e0274c751.pdf
	nx, ny, nz = n[...,0], n[...,1], n[...,2]
	nxs = nx * nx
	nys = ny * ny
	return np.array([
		[0.28209479177387814] * len(nx),	# 0.5 * np.sqrt(1.0 / np.pi)
		0.4886025119029199 * ny, # 0.5 * np.sqrt(3.0 / np.pi)
		0.4886025119029199 * nz,
		0.4886025119029199 * nx,
		1.0925484305920792 * ny * nx, # 0.5 * np.sqrt(15.0 / np.pi)
		1.0925484305920792 * ny * nz,
		0.31539156525252005 * (2.0*nz*nz - nxs - nys), # 0.25 * np.sqrt(5.0 / np.pi)
		1.0925484305920792 * nz * nx, # 0.5 * np.sqrt(15.0 / np.pi)
		1.0925484305920792 * (nxs - nys)
	], np.float32)

def generateAlbedomap(isomap, normmap, iterations=1):
	"""
	generate an delighted albedo map from illuminated texture and normal map
	normals must be normalized and in range [-1,1]
	"""
	isocopy = isomap.copy()
	a = isocopy
	for _ in range(iterations):
		# convert to gray and find average brightness of all facial pixel
		graymap = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_BGR2GRAY)
		cond = np.any(normmap > 0, axis=2)
		a = np.mean(graymap[cond])
		# estimate light coefficents
		Y = generateSphericalHarmonics2(normmap[cond]).T
		k = cv2.solve(a * Y, graymap[cond], flags=cv2.DECOMP_SVD)[1]
		# estimate albedo
		a = isocopy[..., 0:3]
		k = Y @ k
		a[cond] = isocopy[cond, 0:3] / (k * 0.5 + 0.5)
	# isocopy[..., 0:3] = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
	return isocopy

def generateNormalmap(mesh, texture_size):
	"""
	generate a normal map of size (texture_size, texture_size, 4) from eos mesh
	normal map values will be normalized and in range [-1,1] on channels rgb and in range [0,1] on channel a
	"""
	faces = np.asarray(mesh.tvi)
	ver = np.asarray(mesh.vertices)[faces]
	tex = np.asarray(mesh.texcoords)[faces] * texture_size	# mesh.tti is not set but mesh.tvi works fine
	# calculate normal per face
	n = np.cross(ver[:,1] - ver[:,0], ver[:,2] - ver[:,0])
	n /= np.linalg.norm(n, axis=1)[..., None]
	# calculate normal per vertex
	norm = np.zeros((len(mesh.vertices), 3), np.float32)
	norm[faces[:,0]] += n
	norm[faces[:,1]] += n
	norm[faces[:,2]] += n
	norm /= np.linalg.norm(norm, axis=1)[..., None]
	if False:	# normals per pixel
		norm = norm * 0.5 + 0.5
		# render normal map to cairo surface
		import cairo
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
		normmap[..., 0:3] = normmap[..., 0:3] / 127.5 - 1.0
		normmap[..., 3] /= 255
	else:	# normals per face (subdivided into 3 triangles)
		# render normal map
		normmap = np.zeros((texture_size, texture_size, 4), np.float32)
		for t, n in zip(tex, norm[faces]):
			m = np.sum(t, axis=0) / 3
			cv2.fillPoly(normmap, np.array([[t[0], t[1], m]], dtype=np.int32), [*n[0].tolist(), 1])
			cv2.fillPoly(normmap, np.array([[t[1], t[2], m]], dtype=np.int32), [*n[1].tolist(), 1])
			cv2.fillPoly(normmap, np.array([[t[2], t[0], m]], dtype=np.int32), [*n[2].tolist(), 1])
		n = normmap[..., 0:3]
		n = cv2.cvtColor(n, cv2.COLOR_RGB2BGR)
		# blur normal map and normalize again
		n = cv2.blur(n, (texture_size//65,texture_size//65))
		cond = normmap[..., 3] > 0
		normmap[cond, 0:3] = n[cond] / np.linalg.norm(n[cond], axis=1)[...,None]
	return np.flip(normmap, 1)
