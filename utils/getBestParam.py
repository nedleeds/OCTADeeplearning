import os

def getBestParam(args):
	fold_num = args.fold_num
	disease = args.disease.split()
	disease.remove('NORMAL')
	disease = ('_').join(sorted(disease))
	ae = 'ae_o' if args.ae =='True' else 'ae_x'
	best_parameter_dir = os.path.join('./Data/output/best_parameter',disease, args.model, args.filter, ae)
	os.makedirs(best_parameter_dir, exist_ok= True)
	best_param ={f'fold{idx}':{} for idx in range(1,fold_num+1)} 

	best_parameter_path = f'{best_parameter_dir}/best_parameter.txt'
	if not os.path.isfile(best_parameter_path):
		baseline_best_param_path = './Data/output/best_parameter/AMD_CSC_DR_RVO/CV5FC2_3D/OG/ae_x/best_parameter.txt'
		os.system(f'cp {baseline_best_param_path} {best_parameter_path}')

	with open(best_parameter_path) as f:
		keys = f.readline().rstrip().split('\t')[1:]

		for idx in range(1,fold_num+1):
			for key in keys:
				best_param[f'fold{idx}'][key] = ''

		values = 'start'
		for _ in range(fold_num):
			values = f.readline().strip().split('\t')	
			fold_idx = values[0].rstrip()
			for k,v in zip(keys, values[1:]):
				best_param[f'fold{fold_idx}'][k]=v.strip()	

	return best_param
