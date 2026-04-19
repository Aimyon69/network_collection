import torch
from . import model

def export_onnx():
    torch.set_grad_enabled(False)
    device = torch.device('cpu')

    net = model.RetinaFace(cfg=model.cfg_mnet,phase='test').to(device)

    weight_path = './weights/mobilenet0.25_epoch_99.pth'
    state_dict = torch.load(weight_path,map_location=device,weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    dummy_input = torch.randn(1,3,model.cfg_mnet['image_size'][0],model.cfg_mnet['image_size'][1])

    onnx_path = './onnx/retinaface.onnx'

    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params = True,
        opset_version = 11,
        do_constant_folding = True,
        input_names = ['input'],
        output_names = ['cls_prob','bbox_reg','landmark_reg'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'bbox_reg': {0: 'batch_size'},
            'cls_prob': {0: 'batch_size'},
            'landmark_reg': {0: 'batch_size'}
            }
    )

    print('success')

if __name__ == '__main__':
    export_onnx()
