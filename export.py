import argparse
import collections
import os

import torch

import model.model as module_arch
from parse_config import ConfigParser


def main(config: ConfigParser):
    logger = config.get_logger('export')

    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model = model.to('cpu')
    model.eval()

    config.export_dir.mkdir(parents=True, exist_ok=True)

    file_name = 'model.onnx'
    file_path = os.path.join(config.export_dir, file_name)

    x = torch.randn(*config['export']['dummy_input_shape'],
                    requires_grad=True).to('cpu')

    torch.onnx.export(
        model,
        x,
        file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )

    logger.info("Exporting model: {} ...".format(file_path))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
