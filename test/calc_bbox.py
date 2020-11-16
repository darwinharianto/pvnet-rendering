from pvnet_rendering.opengl.opengl_renderer import OpenGLRenderer

renderer = OpenGLRenderer('/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/hsr_ply/fixed_hsr.ply')
print(f"renderer.model['pts'].shape: {renderer.model['pts'].shape}")
print(f"renderer.model['pts']: {renderer.model['pts']}")
xmin = renderer.model['pts'][:, 0].min()
xmax = renderer.model['pts'][:, 0].max()
ymin = renderer.model['pts'][:, 1].min()
ymax = renderer.model['pts'][:, 1].max()
zmin = renderer.model['pts'][:, 2].min()
zmax = renderer.model['pts'][:, 2].max()
print(f'(xmin,xmax): ({xmin},{xmax}), (ymin,ymax): ({ymin},{ymax}), (zmin,zmax): ({zmin},{zmax})')
xlen = xmax - xmin
ylen = ymax - ymin
zlen = zmax - zmin
print(f'xlen: {xlen}, ylen: {ylen}, zlen: {zlen}')