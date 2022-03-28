import os
import json
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import transforms
from src import SSD300, Backbone
from draw_box_utils import draw_box_cycle
import numpy as np
from os import listdir



def create_model(num_classes, num_names=4, is_intu=False, device=torch.device('cpu')):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes, num_names=num_names)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # create model
    num_classes, num_names = 1, 4
    is_intu = False
    model = create_model(num_classes=num_classes+1,
                         num_names=num_names+1,
                         is_intu=is_intu,
                         device=device)

    # load train weights
    train_weights = "./save_weights/ssd.pth"
    train_weights_dict = torch.load(train_weights, map_location=device)
    json_file_box = "./intu_boxConifdences.json"
    json_file_name ="./intu_names.json"
    assert os.path.exists(json_file_box), "{} file not exist.".format(json_file_box)
    assert os.path.exists(json_file_name), "{} file not exist.".format(json_file_name)
    json_file_box = open(json_file_box, 'r')
    json_file_name = open(json_file_name, 'r')
    class_dict_box = json.load(json_file_box)
    class_dict_name = json.load(json_file_name)
    category_index_box = {v: k for k, v in class_dict_box.items()}
    category_index_name = {str(v): k for k, v in class_dict_name.items()}

    # load image for test
    file_names = listdir('./example_predict_data')
    file_names.sort(key=lambda x: int(x[:-4]))
    model.load_state_dict(train_weights_dict['model'], strict=False)
    model.to(device)
    noobject = []
    for tmp_name in file_names:
        original_img = Image.open("./example_predict_data"+tmp_name)
        data_transform = transforms.Compose([transforms.Resize(),
                                             transforms.ToTensor(),
                                             transforms.Normalization()])
        img, _ = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)
        model.eval()
        with torch.no_grad():
            init_img = torch.zeros((1, 3, 300, 300), device=device)
            model(init_img)
            predictions, names = model(img.to(device))
            predictions = predictions[0]
            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()
            if len(predict_boxes):
                predict_boxes = np.asarray([predict_boxes[0]])
                predict_classes = np.array([predict_classes[0]])
                predict_scores = np.array([predict_scores[0]])
            else:
                noobject.append(tmp_name)
            name_id = torch.argmax(names).item()
            if name_id != 0:
                predict_name = category_index_name[str(name_id)]
            if len(predict_boxes) == 0:
                print("No target detected!")
            try:
                draw_box_cycle(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     predict_name,
                     category_index_box,
                     thresh=0.1,
                     line_thickness=5)
                plt.imshow(original_img)
                original_img.save('./result/' + tmp_name)
            except:
                print('Article {} Data processing failure'.format(tmp_name))



if __name__ == "__main__":
    main()
