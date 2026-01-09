
DEFAULT_MODEL_REVISION = None


class Fields(object):
    """ Names for different application fields
    """
    hub = 'hub'
    datasets = 'datasets'
    framework = 'framework'
    cv = 'cv'
    nlp = 'nlp'
    audio = 'audio'
    multi_modal = 'multi-modal'
    science = 'science'
    server = 'server'
class ModeKeys:
    TRAIN = 'train'
    EVAL = 'eval'
    INFERENCE = 'inference'

class ModelFile(object):
    CONFIGURATION = 'configuration.json'
    README = 'README.md'
    TF_SAVED_MODEL_FILE = 'saved_model.pb'
    TF_GRAPH_FILE = 'tf_graph.pb'
    TF_CHECKPOINT_FOLDER = 'tf_ckpts'
    TF_CKPT_PREFIX = 'ckpt-'
    TORCH_MODEL_FILE = 'pytorch_model.pt'
    TORCH_MODEL_BIN_FILE = 'pytorch_model.bin'
    VOCAB_FILE = 'vocab.txt'
    ONNX_MODEL_FILE = 'model.onnx'
    LABEL_MAPPING = 'label_mapping.json'
    TRAIN_OUTPUT_DIR = 'output'
    TRAIN_BEST_OUTPUT_DIR = 'output_best'
    TS_MODEL_FILE = 'model.ts'
    YAML_FILE = 'model.yaml'
    TOKENIZER_FOLDER = 'tokenizer'
    CONFIG = 'config.json'

class ConfigFields(object):
    """ First level keyword in configuration file
    """
    framework = 'framework'
    task = 'task'
    pipeline = 'pipeline'
    model = 'model'
    dataset = 'dataset'
    preprocessor = 'preprocessor'
    train = 'train'
    evaluation = 'evaluation'
    postprocessor = 'postprocessor'

class Frameworks(object):
    tf = 'tensorflow'
    torch = 'pytorch'
    kaldi = 'kaldi'

class Devices:
    """device used for training and inference"""
    cpu = 'cpu'
    gpu = 'gpu'
    
class Invoke(object):
    KEY = 'invoked_by'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'
    