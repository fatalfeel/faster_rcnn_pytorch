from typing import List, Tuple
from config.config import Config

class EvalConfig(Config):

    RPN_PRE_NMS_TOP_N: int  = 6000
    RPN_POST_NMS_TOP_N: int = 300

    ANCHOR_SMOOTH_L1_LOSS_BETA: float   = 1.0
    PROPOSAL_SMOOTH_L1_LOSS_BETA: float = 1.0

    BATCH_SIZE: int             = 1
    LEARNING_RATE: float        = 0.00001
    MOMENTUM: float             = 0.9
    WEIGHT_DECAY: float         = 0.0005
    STEP_LR_SIZES: List[int]    = [50000, 70000]
    STEP_LR_GAMMA: float        = 0.1
    WARM_UP_FACTOR: float       = 0.3333
    WARM_UP_NUM_ITERS: int      = 500

    NUM_STEPS_TO_DISPLAY: int   = 20
    NUM_SAVE_EPOCH_FREQ: int    = 5
    NUM_EPOCH_TO_FINISH: int    = 100

    @classmethod
    def setup(cls, image_min_side: float = None,
              image_max_side: float = None,
              anchor_ratios: List = None,
              anchor_sizes: List = None,
              #pooler_mode: str = None,
              rpn_pre_nms_top_n: int = None,
              rpn_post_nms_top_n: int = None):
        #super().setup(image_min_side, image_max_side, anchor_ratios, anchor_sizes, pooler_mode)
        super().setup(image_min_side, image_max_side, anchor_ratios, anchor_sizes)

        if rpn_pre_nms_top_n is not None:
            cls.RPN_PRE_NMS_TOP_N = rpn_pre_nms_top_n
        if rpn_post_nms_top_n is not None:
            cls.RPN_POST_NMS_TOP_N = rpn_post_nms_top_n
