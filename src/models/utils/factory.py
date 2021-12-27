import logging

logger = logging.getLogger(__name__)

from ..resnet import resnet50, resnet101


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name=='resnet101':
        model = resnet101(model_params,num_classes = model_params['num_classes'])
    elif args.model_name == 'resnet50':
        model = resnet50(model_params,num_classes = model_params['num_classes'])
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model
