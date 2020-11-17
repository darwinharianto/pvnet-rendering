from pvnet_rendering.config.loadable_config import PVNet_Config

cfg = PVNet_Config()
cfg.save_to_path('config.json', overwrite=True)
assert cfg == PVNet_Config.load_from_path('config.json')
assert cfg == PVNet_Config.from_dict(cfg.to_dict())