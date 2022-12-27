#!/usr/bin/env python3

import sys, argparse
from pathlib import Path


FAR_DIR = Path(__file__).resolve().parents[1]
if not str(FAR_DIR) in sys.path:
	sys.path.append(str(FAR_DIR))
import lib.far_fitting as far


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image', nargs='*')
	parser.add_argument('-ls', '--lambda_sh', required=False, default=1000.0)
	parser.add_argument('-le', '--lambda_expr', required=False, default=100000.0)
	parser.add_argument('-o', '--out_dir', required=False, default='result')
	args, _ = parser.parse_known_args()

	# fit avatar
	mesh_name = 'neutral'
	generate_texture = True
	far_pipeline = far.FaceAvatarReconstruction(far.ModelType.FexMM_PCA, far.FittingMode.iterating)
	far_pipeline.reconstructAvatar(args.image, float(args.lambda_sh), float(args.lambda_expr), args.out_dir, mesh_name, generate_texture)
