__version__ = "0.0.0"
import sys
from .src.wrappers import SVMFitter
from .src.wrappers import CLIPProcessor
from .src import model_utils
from .src.trainer import LightWeightTrainer
from .src.clip_utils import get_caption_set
