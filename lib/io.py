import csv
from pathlib import Path

import numpy as np
import cv2
import eos


INPUT_IMAGE_SCALING = 2000	# scales images/landmarks to this heigh. 2000 is recommended


def loadImageAndLandmarksDirectory(image_dir=None):
	images, landmarks = [], []
	if not isinstance(image_dir, list):
		image_dir = [image_dir]
	for p in image_dir:
		p = Path(p)
		if p.is_file():
			files = [p]
		elif p.is_dir():
			files = [f for f in p.iterdir() if f.is_file()]
			files = sorted(files, key=lambda x: str(x))
		for f in files:
			i, l = loadImageAndLandmarksFile(f)
			if l is not None:
				images.append(i)
				landmarks.append(l)
	return images, landmarks

def loadImageAndLandmarksFile(image_path):
	# mainly for internal use. Call loadImageAndLandmarksDirectory instead
	image = landmarks = None
	if cv2.haveImageReader(str(image_path)) and (image_path.parent / f'landmark_coordinates_{image_path.stem}.csv').exists:
		image = cv2.imread(str(image_path))

		if image is not None:
			sf = INPUT_IMAGE_SCALING / image.shape[0]
			try:
				landmarks = np.loadtxt(image_path.parent / f'landmark_coordinates_{image_path.stem}.csv', delimiter=',')
				# img = image.copy()
				# for l in landmarks:
				# 	cv2.circle(img, (int(l[0]), int(l[1])), 2, (127,255,0), -1)
				# saveImage(f'lmks_{image_path.stem}.png', img)
				landmarks = convertLandmarks(np.reshape(landmarks, (-1, 2)).T * sf)
			except IOError:
				pass
			if sf != 1.0:
				image = cv2.resize(image, None, fx=sf, fy=sf)

	return image, landmarks


def convertLandmarks(landmarks_raw, scale=1):
	"""A helper function to calculate the 68 ibug landmarks from an image."""
	return list(map(lambda i, x, y: eos.core.Landmark(str(i + 1), [x * scale, y * scale]), range(len(landmarks_raw[0])), *landmarks_raw))


def loadDict(path):
	d = {}
	if path.exists():
		with path as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			next(reader, None)
			for row in reader:
				d.update({
					row[0]: {
						row[1]: {
							"identity_coeffs": eval(row[2]),
							"expression_coeffs": eval(row[3])
						}
					}
				})
	return d

def updateDict(path, j, faceID, mesh_name, d):
	if faceID in j:
		j[faceID][mesh_name] = d
	else:
		j[faceID] = {mesh_name: d}
	with open(path, 'w') as outfile:
		writer = csv.DictWriter(outfile, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_MINIMAL, fieldnames=["Face ID", "Mesh Name", *d.keys()])
		writer.writeheader()
		for k, v in j.items():
			for kk, vv in v.items():
				d = {"Face ID": k, "Mesh Name": kk}
				d.update(vv)
				writer.writerow(d)


def saveMesh(path, mesh):
	eos.core.write_textured_obj(mesh, str(path))


def saveImage(path, image):
    cv2.imwrite(str(path), image)
