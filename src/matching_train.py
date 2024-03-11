import boto3
import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_loader import RoofTopImage
from backbone import R2U_Net, NonMaxSuppression, DetectionBranch
from matching import AttentionalGNN,ScoreNet,OptimalMatching
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss 
from torch.optim import Adam
from torchsummary import summary
import itertools
import gc
import json
import time
from utils import scores_to_permutations,permutations_to_polygons,permutation_mask
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import losses as ls
from shapely.geometry import Polygon
import rasterio
from shapely.validation import explain_validity
from kdtree import kdtree

IMAGE_BUCKET="sundial-geometric-roof-inference"
VAL_IMAGE_PREFIX= "images/val" 
TRAIN_IMAGE_PREFIX= "images/train"

TRAIN_ANNOTATION_PATH = '../gt_annotations/raw_train_annotations.json'
VAL_ANNOTATION_PATH = '../gt_annotations/raw_val_annotations.json'

PREDICTION_PATH = 'raw_val_predictions.json'

BATCH_SIZE = 6
MAX_CORNER_POINTS = 50
NUM_EPOCHS = 70
INIT_LR = 0.001
LAMBDA = 1000
IMG_SIZE = 320
BASE_OUTPUT = "output"
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "train_val_loss_matching_pretrain_run50_AVGls_70_ephoc.png"])
TRAINING_OUTPUT = os.path.sep.join([BASE_OUTPUT, 'train_val_loss_matching_pretrain_run50_AVGls_70_ephoc.json'])
MATCHING_MODEL_PATH = os.path.join(BASE_OUTPUT+'/weights', "matching_pretrain_run50_AVGls_70_ephoc.pth")


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    gc.collect()
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

torch.autograd.set_detect_anomaly(True) # for debugging purpose

print(MATCHING_MODEL_PATH)
print(TRAINING_OUTPUT)
print(PLOT_PATH)


cnn_backbone = R2U_Net()
cnn_backbone = cnn_backbone.to(device)
cnn_backbone = cnn_backbone.train()

conv_one_one = DetectionBranch()
conv_one_one = conv_one_one.to(device)
conv_one_one = conv_one_one.train()

cnn_backbone.load_state_dict(torch.load('../trained_weights/polyworld_backbone',map_location=device)) #transfer learning
conv_one_one.load_state_dict(torch.load("../trained_weights/polyworld_seg_head",map_location=device)) #transfer learning

suppression = NonMaxSuppression(MAX_CORNER_POINTS)
suppression = suppression.cuda()

matching = OptimalMatching()
matching = matching.cuda()
matching.load_state_dict(torch.load("../trained_weights/polyworld_matching",map_location=device))
#matching.load_state_dict(torch.load("output/weights/matching_pretrain_run2.pth",map_location=device))


#freeze the backbone and vertex detection neworks 
for param in cnn_backbone.parameters():
    param.requires_grad = False
for param in conv_one_one.parameters():
    param.requires_grad = False
    
train_image_s3_path = "../data/train"#"s3://{}/{}".format(IMAGE_BUCKET,TRAIN_IMAGE_PREFIX)
val_image_s3_path = "../data/val" #"s3://{}/{}".format(IMAGE_BUCKET,VAL_IMAGE_PREFIX)


train_data = RoofTopImage(train_image_s3_path,
                          TRAIN_ANNOTATION_PATH,
                          MAX_CORNER_POINTS, 
                          IMG_SIZE,BATCH_SIZE)

trainloader = DataLoader(train_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=os.cpu_count(),
                        pin_memory=False,
                        prefetch_factor= BATCH_SIZE
                    )

val_data = RoofTopImage(val_image_s3_path,
                        VAL_ANNOTATION_PATH,
                        MAX_CORNER_POINTS,
                        IMG_SIZE,BATCH_SIZE)


valloader = DataLoader(val_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=os.cpu_count(),
                        pin_memory=False,
                        prefetch_factor= BATCH_SIZE
                    )

train_coco = train_data.img_annotations
val_coco = val_data.img_annotations

#pos_weight = torch.tensor([100]).to(device)
#lossFunc = BCEWithLogitsLoss(pos_weight = pos_weight)

#params = [matching.parameters(),conv_one_one.parameters()]
#optimizer = Adam(itertools.chain(*params), lr = INIT_LR)

optimizer = Adam(matching.parameters(), lr = INIT_LR)

#calculate steps per epoch for training and test set
trainSteps = len(train_data) // BATCH_SIZE
valSteps = len(val_data) // BATCH_SIZE

print('Training and validation steps {} {}'. format(trainSteps,valSteps))

H= {"train_loss": [], "test_loss": [] } #to keep training history


def nearest_neighboor_2d(tree, data, gt_points):
    cndd_gt = {(k[0], k[1]):-1 for k in data}
    gt_cndd = {}
    
    max_corner_points = sys.maxsize
    closest_set = set()
    for gt_pt in gt_points:
        for gt_point in zip(gt_pt[::2], gt_pt[1::2]):
            found = False
            while not found:
                nearest_node = tree.find_nearest_dist_node(gt_point, max_corner_points).data 
                if tuple(nearest_node) not in closest_set:
                    closest_set.add(tuple(nearest_node))
                    found = True 
                else:
                    tree.remove(nearest_node)

            if found:
                gt_cndd[gt_point] = tuple(nearest_node)
                cndd_gt[tuple(nearest_node)] = gt_point                 
    
    return cndd_gt, gt_cndd
            
def match_nearest_point(nms_points,anns,gt_index):
    #n, d = nms_points.shape
    sorted_points = sorted(nms_points, key=lambda k: [k[0], k[1]]) #sort nms points sorted(nms_points[b] , key=lambda k: [k[0], k[1]])
    n = len(sorted_points)
    #sorted_points = sorted_points.detach().cpu().numpy()
    #new_nms_points = np.empty_like(sorted_points, np.int8)
    #orderd_index = {tuple(v):i for i, v in enumerate(sorted_points)} # keep track of indices in the sorted nms points 
   
    tree = kdtree(sorted_points) # create a kd tree with sorted nms points
    #find the nearest ground truth points 
    gt_points = [ann['segmentation'][0] for ann in anns]
    cndd_gt, gt_cndd =nearest_neighboor_2d(tree, sorted_points, gt_points) #gt_index is the clockwise ordering of ground truth points
    
    selected_c_points = list(gt_cndd.values())
    matched_gt_points = list(gt_cndd.keys())
    gt_keys = list(gt_index.keys())
    '''
    print(nms_points)
    print()
    print(matched_gt_points)
    print()
    print(gt_keys)
    '''
    assert len(matched_gt_points) == len(gt_keys)
    assert len(selected_c_points) == len(set(selected_c_points))
    
    #now ensuring to have index consitency between candidate points and the ground truth permutation matrix, gt_perm
    for k, v in gt_index.items():
        sorted_points[v] = gt_cndd[k]
        
    restart_index = list(gt_index.values())[-1]+1
    
    for k, v in cndd_gt.items():
        if v == -1:
            sorted_points[restart_index] = list(k)
            restart_index +=1
            if restart_index == n:
                break
    
    #print('length of updated nms is {} and {}'.format(len(updated_nms), len(updated_nms[0])))
    if tree:
        del tree
    return sorted_points

def permutation_metrix(idx, nms_points, coco):
    B, _, D= nms_points.shape
    perm_mask = np.empty([B, MAX_CORNER_POINTS, MAX_CORNER_POINTS ])
    
    nms_points = nms_points.detach().cpu().numpy()
    updated_nms_points = np.empty_like (nms_points)
    
    for im_indx, img_id in enumerate(idx): # for each instance in the batch
        img = coco.loadImgs([img_id.item()])[0]
        annids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annids)
        
        gt_perm, gt_index = permutation_mask(anns,p_size=MAX_CORNER_POINTS)
     
        #get nearest points
        new_nms_points = match_nearest_point(nms_points[im_indx].tolist(), anns,gt_index)
        updated_nms_points[im_indx] = new_nms_points
        
        gt_perm[range(list(gt_index.values())[-1]+1, MAX_CORNER_POINTS), range(list(gt_index.values())[-1]+1, MAX_CORNER_POINTS)] = 1 #filling diagonals with 1
        perm_mask[im_indx] = gt_perm.reshape(1, MAX_CORNER_POINTS ,MAX_CORNER_POINTS )
        
    #finally return the ground truth permutation matrix and updated nms points
    return torch.from_numpy(perm_mask),torch.from_numpy(updated_nms_points)
    
def create_polygon(pp):
    points = [[pp[i*2], pp[i*2+1]] for i in range(len(pp) // 2)]
    return Polygon(points)

def get_polygons(pred_poly,segmentations):
    pred_polygons = []
    gt_polygons = []
    for i , pp in enumerate(pred_poly):
        p_p = []
        for p in pp:
            p_p.append(create_polygon(p))
        pred_polygons.append(p_p)
        
    for _, ann in enumerate(segmentations):
        gt_pp = []
        for att in ann:
            p = [tensor[0].item() for tensor in att['segmentation'][0]]
            gt_pp.append(create_polygon(p))
        gt_polygons.append(gt_pp)
        
    return pred_polygons, gt_polygons
#equation 12
def angle_loss(pred_polygons,gt_polygons):
    overlap = False
    a_loss = 0.0
    for gt_p in gt_polygons:
        for pr_p in pred_polygons:
            if gt_p.intersects(pr_p): # check if the two polygons overlap
                overlap == True
                #print('groudtruth ', gt_p)
                #print('predicted polygo ', pr_p)
                #calculate anlge loss
                a_loss +=ls.angle_loss(gt_p, pr_p)
        #calculate angle loss
        if not overlap:
            a_loss +=ls.angle_loss(gt_p)
    return a_loss

def polygon_mask(pred_poly):
    K = len(pred_poly)
    poly_mask = np.empty([K,IMG_SIZE, IMG_SIZE])
    for k, p in enumerate(pred_poly): # for each predicted polygon 
        arr_x, arr_y= p.exterior.xy
        con = torch.column_stack((torch.tensor(arr_x), torch.tensor(arr_y)))
        x = con.unfold(dimension = 0,size = 2, step = 1)
        m = np.empty([IMG_SIZE, IMG_SIZE])
        for row in range(IMG_SIZE):
            x_i = torch.tensor([row for _ in range(IMG_SIZE)])
            y_j = torch.tensor([i for i in range(IMG_SIZE)])
            indices = torch.column_stack((x_i, y_j))
            for ind in indices: # for each pixel location
                mat = []
                for v_ind in range(x.shape[0]): # for each consecutive vertices u and v of polygon k
                    u_v = x[v_ind].T
                    l1 = [(u_v[0][0], u_v[0][1]),(ind[0], ind[1])] # between u and ind
                    l2 = [(ind[0], ind[1]), (u_v[1][0], u_v[1][1])] # between v and ind
                    
                    ang = ls.calculate_angle(l1,l2)
                
                    t = torch.cat((x[v_ind], ind.reshape(2,1)), dim =1)
                    ones = torch.ones(3).reshape(1,3)
                    t = torch.cat((t, ones))
                    mat.append(t)
                    #print(t)
                dets = torch.mul(torch.linalg.det((torch.stack(mat, dim=0))), LAMBDA)
                #print(dets)
                dets_ = torch.add(torch.abs(dets),1)
                m[ind[0]][ind[1]] = torch.sum(torch.mul(torch.div(dets,dets_), ang))
        poly_mask[k] = m 
        
    return poly_mask


c = 0

startTime = time.time()
best_val_loss = sys.float_info.max
for e in tqdm(range(NUM_EPOCHS)):
    matching.train()
    totaltrainloss, totalvalloss =0,0
    
    for i, train_batch in enumerate(trainloader):
        (x, img_indx) =(train_batch['image'].to(device),
                        train_batch['image_idx'])

        feature_map = cnn_backbone(x)
        vertex_mask = conv_one_one(feature_map)
        
        #print ('feature map shape', feature_map.shape)
        #print('vertex_mask shape', vertex_mask.shape)
        _, nms_graph = suppression(vertex_mask)
        
        #nms_graph, _= torch.sort(nms_graph, dim=-1) #sort candidate corner points before using them to extract descriptors from the feature map
        #print('NMS graph shape', nms_graph.is_cuda)
        #print(nms_graph)
        #print(img_indx)
        #print(type(img_indx))
        
        gt_perumation_mask, updated_nms_points = permutation_metrix(img_indx, 
                                                                    nms_graph, 
                                                                    train_coco
                                                                   )
        
        partial_assignment= matching(x, 
                                      feature_map, 
                                      updated_nms_points.to(device)
                                      ) 
        
        #get L_match equation 11
        l_match = ls.cross_entropy_loss(gt_perumation_mask.to(device), partial_assignment)
        optimizer.zero_grad()
        l_match.backward()
        optimizer.step()
        
        #angle loss equation 12 l_angle
        #pred_polygons, gt_polygons = get_polygons(pred_polys,train_coco)
        #l_angle = 0.0
        #for b in range(BATCH_SIZE):
            #l_angle += angle_loss(pred_polygons[b],gt_polygons[b])

        #print('angle loss is ', l_angle)

        # segmentation loss l_seg
        #for b in range(BATCH_SIZE):
            #print(polygon_mask(pred_polygons[b]).shape)
 
        #add the loss to the total training loss so far
        totaltrainloss += float(l_match)
            
        #if i % 100 == 0:
            #print(" Finish the iter : {}, loss {}".format(i, float(l_match)))

        del x
        del img_indx
        del feature_map
        del vertex_mask
        del nms_graph
        del updated_nms_points
        del gt_perumation_mask
        del partial_assignment
        

    with torch.no_grad():
        matching.eval()
        
        for val_batch in valloader:
            (y, val_img_indx) =(
                                 val_batch['image'].to(device),
                                 val_batch['image_idx']
                                 ) 
            pred_val = cnn_backbone(y)
            vertex_mask_val = conv_one_one(pred_val)
            
            _, val_nms_graph = suppression(vertex_mask_val)
            
            val_gt_perumation_mask, val_updated_nms_points = permutation_metrix(val_img_indx, 
                                                                                val_nms_graph, 
                                                                                val_coco)
            val_partial_assignment = matching(y, 
                                              pred_val, 
                                              val_updated_nms_points.to(device))

            #get L_match equation 11
            l_val_match = ls.cross_entropy_loss(val_gt_perumation_mask.to(device), val_partial_assignment)
            totalvalloss += float(l_val_match)
            
            del y
            del pred_val
            del val_img_indx
            del vertex_mask_val
            del val_nms_graph
            del val_gt_perumation_mask
            del val_updated_nms_points
            del val_partial_assignment
            
    #calculate the average training and validation loss
    avgTrainLoss = totaltrainloss / trainSteps
    avgValLoss = totalvalloss  / valSteps
    
    if best_val_loss > avgValLoss:
        print('Got a new best val loss. Saving model ...')
        best_val_loss = avgValLoss
        torch.save(matching.state_dict(), MATCHING_MODEL_PATH)
    
    H['train_loss'].append(avgTrainLoss)
    H['test_loss'].append(avgValLoss)
   
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgValLoss))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

with open( TRAINING_OUTPUT, 'w') as f:
    json.dump(H, f)