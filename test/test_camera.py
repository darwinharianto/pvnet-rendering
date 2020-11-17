from annotation_utils.linemod.objects import LinemodCamera

camera = LinemodCamera.from_image_shape([1080, 1920, 3])
print(camera)
print(camera.to_matrix())
camera.save_to_txt('camera.txt', overwrite=True)
assert camera == LinemodCamera.load_from_txt('camera.txt')
assert camera == LinemodCamera.from_matrix(camera.to_matrix())
assert camera == LinemodCamera.from_dict(camera.to_dict())