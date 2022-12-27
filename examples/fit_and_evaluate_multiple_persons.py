#!/usr/bin/env python3

import sys
import csv
import time
from pathlib import Path
from multiprocessing import Pool as mp_Pool

FAR_DIR = Path(__file__).resolve().parents[1]
if not str(FAR_DIR) in sys.path:
	sys.path.append(str(FAR_DIR))
import examples.evaluation as evaluation
import lib.far_fitting as far


PARALLEL_PROCESSES = 12
MORPHABLE_MODEL = far.ModelType.FexMM_PCA	# valid values: FexMM-AU, FexMM-PCA, FexMM-NN, SFM, BFM, Facegen-Model, Flame
FITTING_MODE = far.FittingMode.iterating	# must be iterating if expressions are pca model. Can be collective if expressions are blendshapes
evaluation.USE_MANUAL_ANTHROPOMETRIC_LANDMARKS = True	# only ULM data base has manual marked anthropometric landmarks
evaluation.SAVE_REGISTERED_MESHS = False


def parallelRunFitting(in_dir, out_dir, ls, le):
	# for intern use (parallelization) only.
	try:
		parallelRunFitting.far_pipeline
	except AttributeError:
		parallelRunFitting.far_pipeline = far.FaceAvatarReconstruction(MORPHABLE_MODEL, FITTING_MODE)
	parallelRunFitting.far_pipeline.reconstructAvatar([in_dir], ls, le, out_dir, 'neutral', False)

def parallelRunEvaluation(i, e, fitted_dir, reference_dir):
	# for intern use (parallelization) only.
	try:
		parallelRunEvaluation.test
	except AttributeError:
		parallelRunEvaluation.test = evaluation.TestBed(MORPHABLE_MODEL)
	return i, e, *parallelRunEvaluation.test.evaluate(e, fitted_dir, reference_dir)

def buildModelName():
	mm = {
		'FexMM_PCA': 'zfd-pca_pca',
		'FexMM_NN': 'zfd-nn_bl',
		'FexMM_AU': 'zfd-au_pca',
		'BFM': 'bfm_pca',
		'FaceGen': 'fgm_pca',
		'FLAME': 'flm_pca'
	}
	return f"{mm[MORPHABLE_MODEL.name]}_{'single' if FITTING_MODE == far.FittingMode.iterating else 'multi'}"

def updateDiffCSV(CSV_FILE, model_name, ls, le, ret):
	CSV_FILE.parents[0].mkdir(parents=True, exist_ok=True)
	if not CSV_FILE.exists():
		with open(str(CSV_FILE), 'w') as outfile:
			s = 'Fitted Model, Lambda Identity, Lambda Expression, Face ID, Expression, RMSE selected Landmarks, RMSE Mesh, Min Distance Mesh, Max Distance Mesh'
			landmarks = list(zip(*ret))[6]
			if landmarks is not None:
				s += ', ' + ', '.join(landmarks[0].keys())
			outfile.write(s + '\n')
	with open(CSV_FILE, 'a') as outfile:
		outfile.write(f'\n')
	for i, e, rmse_l, rmse_m, dmin, dmax, landmarks in ret:
		if landmarks is None:
			landmarks = ''
		else:
			landmarks = ', ' + ', '.join(map(str, landmarks.values()))
		with open(CSV_FILE, 'a') as outfile:
			outfile.write(f'{model_name}, {ls}, {le}, {i}, {e}, {rmse_l}, {rmse_m}, {dmin}, {dmax}{landmarks}\n')

def updateDataCSV(DATA_FILE, INDIR, name, ls, le, ids, exprs):
	if DATA_FILE.exists():
		with open(DATA_FILE, 'a') as outfile:
			outfile.write('\n')
	for i in ids:
		for e in exprs:
			old_file = INDIR / f'{i.strip()}/data.csv'
			with open(old_file, 'r') as data_file:
				for d in csv.DictReader(data_file, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_MINIMAL):
					dat = {	'Fitted Model': name,
							'Lambda Identity': ls,
							'Lambda Expression': le,
							'Face ID': i,
							'Mesh Name': e }
					for j, s in enumerate(eval(d['identity_coeffs'])):
						dat[f'coeff_id_{j}'] = s
					for j, s in enumerate(eval(d['expression_coeffs'])):
						dat[f'coeff_exp_{j}'] = s
					write_header = not DATA_FILE.exists()
					with open(DATA_FILE, 'a') as outfile:
						if write_header:
							outfile.write(", ".join(map(str, dat.keys())) + "\n\n")
						outfile.write(", ".join(map(str, dat.values())) + "\n")
			old_file.unlink()


def tuneEOS():
	name = buildModelName()
	INPUT_DIR = Path('face_data/Ulm/images_openface')
	OUTPUT_DIR = Path(f'tune_ulm_openface/{name}/{name}')
	REFERENCE_DIR = INPUT_DIR
	CSV_FILE = OUTPUT_DIR.parents[0] / 'diff.csv'
	DATA_FILE = OUTPUT_DIR.parents[0] / 'data.csv'

	ids = [f.name for f in filter(Path.is_dir, INPUT_DIR.iterdir())][:6]
	exprs = ['neutral']
	sh_lambdas = [1000, 2000]
	expr_lambdas = [100000]
	with mp_Pool(PARALLEL_PROCESSES) as pool:
		for le in expr_lambdas:
			for ls in sh_lambdas:
				para = ((INPUT_DIR/i/e, Path(f'{OUTPUT_DIR}_{ls}_{le}')/i, ls, le) for i in ids for e in exprs)
				pool.starmap(parallelRunFitting, para)
				para = ((i, e, Path(f'{OUTPUT_DIR}_{ls}_{le}')/i, REFERENCE_DIR/i/e) for i in ids for e in exprs)
				ret = pool.starmap(parallelRunEvaluation, para)
				updateDiffCSV(CSV_FILE, name, ls, le, ret)
				updateDataCSV(DATA_FILE, Path(f'{OUTPUT_DIR}_{ls}_{le}'), name, ls, le, ids, exprs)


def evaluateFacegenModeller():
	INPUT_DIR = Path('face_data/study_faces/images_openface')
	OUTPUT_DIR = Path('FaceGen')
	REFERENCE_DIR = INPUT_DIR
	CSV_FILE = OUTPUT_DIR / 'diff.csv'

	ids = ['3D1001', '3D1009', '3D1011', '3D1012', '3D1017', '3D1019']
	exprs = ['neutral']
	ret = []
	for i in ids:
		for e in exprs:
			test = evaluation.TestBed(far.ModelType.FaceGen)
			ret.append([i, e, *test.evaluate(f'{i}_{e}', OUTPUT_DIR/'neutral_model'/i, REFERENCE_DIR/i/e)])
	updateDiffCSV(CSV_FILE, 'FaceGen-Program', 0, 0, ret)


def evaluateFLAME_TF():
	INPUT_DIR = Path('face_data/ulm/images_openface')
	OUTPUT_DIR = Path('Flame_tf/ulm')
	REFERENCE_DIR = INPUT_DIR
	CSV_FILE = OUTPUT_DIR / 'diff.csv'

	ids = [f.name for f in filter(Path.is_dir, INPUT_DIR.iterdir())]
	ret = []
	for i in ids:
		for e in [f.stem for f in (OUTPUT_DIR/i).rglob('*.obj')]:
			test = evaluation.TestBed(far.ModelType.FLAME)
			ret.append([i, e, *test.evaluate(e, OUTPUT_DIR/i, REFERENCE_DIR/i/'neutral')])
	updateDiffCSV(CSV_FILE, 'FLAME-Tensorflow', 0, 0, ret)


if __name__ == '__main__':
	t = time.time()
	tuneEOS()
#	evaluateFacegenModeller()
#	evaluateFLAME_TF()
	t = time.time() - t
	print(f'time: {int(t//60)}m {int(t%60):02d}s')

