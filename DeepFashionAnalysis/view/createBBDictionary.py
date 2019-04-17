import re
import cv2
import json
splitter = re.compile("\s+")
def get_dict_bboxes():
    with open('../data/Anno/list_category_img.txt', 'r') as category_img_file, \
            open('../data/Anno/list_eval_partition.txt', 'r') as eval_partition_file, \
            open('../data/Anno/list_bbox.txt', 'r') as bbox_file:
        list_category_img = [line.rstrip('\n') for line in category_img_file][2:]
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_bbox = [line.rstrip('\n') for line in bbox_file][2:]

        list_category_img = [splitter.split(line) for line in list_category_img]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_bbox = [splitter.split(line) for line in list_bbox]

        list_all = [(k[0], k[0].split('/')[1].split('_')[-1], v[1], (int(b[1]), int(b[2]), int(b[3]), int(b[4])))
                    for k, v, b in zip(list_category_img, list_eval_partition, list_bbox)]

        list_all.sort(key=lambda x: x[1])

        dict_train = create_dict_bboxes(list_all,split='train')
        dict_val = create_dict_bboxes(list_all, split='val')
        dict_test = create_dict_bboxes(list_all, split='test')

        return dict_train, dict_val, dict_test

def create_dict_bboxes(list_all, split='train'):
    lst = [(line[0], line[1], line[3], line[2]) for line in list_all if line[2] == split]
    lst = [("".join(line[3] + '/' + line[1] + line[0][3:]), line[1], line[2]) for line in lst]
    #lst_shape = [cv2.imread('../data/' + line[0]).shape for line in lst]
    lst_shape = [cv2.imread('../data/Img/' + '/'.join((line[0].split('/'))[2:])).shape for line in lst]
    lst = [(line[0], line[1], (round(line[2][0] / shape[1], 2), 
                               round(line[2][1] / shape[0], 2), 
                               round(line[2][2] / shape[1], 2), 
                               round(line[2][3] / shape[0], 2))) for line, shape in zip(lst, lst_shape)]
    
    dict_ = {"/".join(line[0].split('/')[1:]): {'x1': line[2][0], 
                                                'y1': line[2][1], 
                                                'x2': line[2][2], 
                                                'y2': line[2][3]} for line in lst}
    return dict_

if __name__ == '__main__':
    dict_train, dict_val, dict_test = get_dict_bboxes()
    with open('dict_train.json','w') as f_train, \
         open('dict_test.json', 'w') as f_test, \
         open('dict_val.json','w') as f_val:
         print(json.dumps(dict_train, indent=4, sort_keys=True),file=f_train)
         print(json.dumps(dict_test , indent=4, sort_keys=True),file=f_test)
         print(json.dumps(dict_val  , indent=4, sort_keys=True),file=f_val)