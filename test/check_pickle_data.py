from pvnet_rendering.util.base_utils import read_pickle
from common_utils.path_utils import get_all_files_of_extension

renders_dir = 'renders0/hsr'
pkl_paths = get_all_files_of_extension(renders_dir, 'pkl')
pkl_paths.sort()

for pkl_path in pkl_paths:
    data = read_pickle(pkl_path)
    print(data.keys())
    RT = data['RT']
    K = data['K']
    print(f'RT: {RT}')
    print(f'K: {K}')