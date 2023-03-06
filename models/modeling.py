import torch 
import torchvision 
import models.deeplabv3plus as deeplabv3plus
from models.ddrnet.ddernet import DDRNet
from models.deeplabv3plus.utils import set_bn_momentum
from models.deeplabv3plus._deeplab import convert_to_separable_conv
from models.rtformer.rtformer import RTFormer
# def get_model(model_name, weights, weights_backbone, num_classes, aux_loss):
#     model = torchvision.models.get_model(
#             model_name,
#             weights=weights,
#             weights_backbone=weights_backbone,
#             num_classes=num_classes,
#             aux_loss=aux_loss,
#         )
    
#     return model

def get_model(model_name, num_classes, weights=None, weights_backbone=None, aux_loss=False):
    if 'plus' in model_name:
        # FIXME: Need to take it into params.
        # https://github.com/VainF/DeepLabV3Plus-Pytorch
        separable_conv = False
        output_stride = 8 # 8 or 16
        model = deeplabv3plus.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
        if separable_conv and 'plus' in model_name:
            convert_to_separable_conv(model.classifier)
        set_bn_momentum(model.backbone, momentum=0.01)
            
    elif model_name == 'ddrnet':
        model = DDRNet(num_classes=num_classes)
        
    else:
        model = torchvision.models.segmentation.__dict__[model_name](pretrained=True, aux_loss=aux_loss)
        if 'fcn' in model_name:
            if model_name.split('_')[1] == 'resnet50' or model_name.split('_')[1] == 'resnet101':
                model.classifier = torchvision.models.segmentation.fcn.FCNHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            else:
                raise ValueError(f"There is no such backbone({model_name.split('_')[1]}) for {model_name}")
        elif model_name.split('_')[0] == 'deeplabv3':
            model = torchvision.models.segmentation.__dict__[model_name](pretrained=True, aux_loss=aux_loss)
            if model_name.split('_')[1] == 'resnet101': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif model_name.split('_')[1] == 'resnet50': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif model_name.split('_')[1] == 'mobilenet': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(10, num_classes, kernel_size=(1, 1), stride=(1, 1))
            else:
                raise ValueError(f"There is no such model({model_name})")
    
    return model

if __name__ == '__main__':
    # model = get_model('deeplabv3plus_resnet101', 3)
    
