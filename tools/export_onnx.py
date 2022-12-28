import numpy as np
import cv2
import onnxruntime
import torch
from models import FaceModel, resnet18, FPN, DetectHead
import albumentations as A
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

def load_model(model,model_path):
    model_dict = torch.load(model_path)['state_dict']
    model.load_state_dict(model_dict)
    return model
def build_model(wide):
    backbone = resnet18()
    neck = FPN(wide=wide)
    head = DetectHead(wide=wide,deploy=True)
    model = FaceModel(backbone, neck, head)
    return model


def transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W):
    model = build_model(24)
    model = load_model(model,weight_file)
    model.eval()

    input_names = ["input"]
    output_names = ["score"]

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "resnet18-1_3_{}_{}_dynamic.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
        dynamic_axes = {"input": {0: "batch_size"}, "score": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "resnet18_{}_3_{}_{}_static.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name,model
def process(image):
    image/=255.0
    image-=mean
    image/=std

    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image,0)
    image = torch.from_numpy(image)
    return image


def main(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W):
    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W)
    else:
        # Transform to onnx as specified batch size
        # transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
        # Transform to onnx for demo
        onnx_path_demo,model = transform_to_onnx(weight_file, 1, IN_IMAGE_H, IN_IMAGE_W)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load('BisNet_shuff_1_3_360_640_static.onnx')
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread('/mnt/e/workspace/FaceDetection/widerface/val/images/0--Parade/0_Parade_marchingband_1_353.jpg')
    image_show = cv2.resize(image_src, (800, 800)).astype(np.float32)
    image_tens = process(image_show)
    with torch.no_grad():
        model_output = model(image_tens)
    ort_inputs = {session.get_inputs()[0].name: image_tens.numpy()}
    result = session.run( [session.get_outputs()[0].name],ort_inputs)
    print(result[0].shape)
    print(np.sum(model_output.numpy()-result[0]))




if __name__ == '__main__':
    model_path = "../train_mode/conventional_training/log/Epoch_44_batch_2999.pt"
    batch_size = 1
    IN_IMAGE_H=800
    IN_IMAGE_W = 800

    main(model_path,batch_size,IN_IMAGE_H,IN_IMAGE_W)