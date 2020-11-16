from pvnet_rendering.util.camera import BasicCamera

camera = BasicCamera.from_image_shape([1080, 1920, 3])
print(camera)
print(camera.to_matrix())
camera.save_to_txt('camera.txt', overwrite=True)
assert camera == BasicCamera.load_from_txt('camera.txt')
assert camera == BasicCamera.from_matrix(camera.to_matrix())
assert camera == BasicCamera.from_dict(camera.to_dict())