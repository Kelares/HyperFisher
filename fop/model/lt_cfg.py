from types import SimpleNamespace

cfg = SimpleNamespace()

# Image preprocessing
cfg.INPUT_SIZE = (32, 32)
cfg.COLOR_SPACE = 'RGB'

# Dataset configuration
cfg.DATASET = SimpleNamespace()
cfg.DATASET.DATASET = 'IMBALANCECIFAR10'
cfg.DATASET.IMBALANCECIFAR = SimpleNamespace()
cfg.DATASET.IMBALANCECIFAR.RATIO = 0.02  # imbalance ratio as per your config
cfg.DATASET.IMBALANCECIFAR.RANDOM_SEED = 0
cfg.DATASET.USE_CAM_BASED_DATASET = False
cfg.DATASET.CAM_DATA_JSON_SAVE_PATH = ''

# Sampler
cfg.TRAIN = SimpleNamespace()
cfg.TRAIN.SAMPLER = SimpleNamespace()
cfg.TRAIN.SAMPLER.TYPE = 'default'  # Can also be 'weighted sampler'
cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER = SimpleNamespace()
cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE = 'balance'  # ignored if TYPE is default

# Two-stage setup (not used, but class expects it)
cfg.TRAIN.TWO_STAGE = SimpleNamespace()
cfg.TRAIN.TWO_STAGE.DRS = False
cfg.TRAIN.TWO_STAGE.START_EPOCH = 0

# Needed if progressive sampler is used
cfg.TRAIN.MAX_EPOCH = 100

cfg.NETWORK = SimpleNamespace()
cfg.NETWORK.PRETRAINED = False
cfg.NETWORK.PRETRAINED_MODEL = ''

cfg.BACKBONE = SimpleNamespace()
cfg.BACKBONE.TYPE = 'res32_cifar'
cfg.BACKBONE.PRETRAINED_MODEL = ""
# Module
cfg.MODULE = SimpleNamespace()
cfg.MODULE.TYPE = 'GAP'

# Classifier
cfg.CLASSIFIER = SimpleNamespace()
cfg.CLASSIFIER.TYPE = 'FC'
cfg.CLASSIFIER.BIAS = True


cfg.RESUME_MODEL = ""
cfg.RESUME_MODE = "all"