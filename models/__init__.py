from .p2pnet import build
from .StudentNet import builded

# build the P2PNet model
# set training to 'True' during training
def build_t(args, training=False):
    return build(args, training)

def build_s(args, training=False):
    return builded(args, training)