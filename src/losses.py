import torch
import torch.nn.functional as F
import math
def iou_loss(pred_mask, target_maks):
    pred_mask = F.sigmoid(pred_mask)
    intersection = (pred_mask * target_maks).sum()
    union = ((pred_mask + target_maks) - (pred_mask * target_maks)).sum()
    iou = intersection / union
    iou_dual = pred_mask.size(0) - iou

    #iou_dual = iou_dual / pred_mask.size(0)
    iou_dual.requires_grad = True
    return torch.mean(iou_dual)
def cross_entropy_loss(sinkhorn_results,gt_permutation):
    #return -1 * (torch.sum(torch.mul(gt_permutation, sinkorn_results)) / b)
    loss_match = -torch.mean(torch.masked_select(sinkhorn_results, gt_permutation == 1))
    return loss_match 
    #return torch.mean(torch.sum(gt_permutation.view(batch_size, -1) * sinkorn_results.view(batch_size, -1), dim=1))

#https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return torch.tensor(ang + 360, requires_grad=True) if ang < 0 else torch.tensor(ang,requires_grad=True)

# dot product
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
#calculate angle
def calculate_angle(l1, l2):
    #print(l1, '->', l2)
    vA = [(l1[0][0]-l1[1][0]), (l1[0][1]-l1[1][1])]
    vB = [(l2[0][0]-l2[1][0]), (l2[0][1]-l2[1][1])]
    #get dot prod
    dot_prd = dot (vA, vB)
    #get magnitude
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5

    r = dot_prd / magA / magB
    r =1 if r > 1 else r
    try:
        angle = math.acos(r)
    except:
        print(r)
        return torch.tensor(0)
    degree_angle = math.degrees(angle) % 360
    degree_angle = 360 - degree_angle if degree_angle >= 180 else degree_angle
    #print('degree angle is ', degree_angle)
    return torch.tensor(degree_angle, requires_grad=True)
    
def angle_loss(p1, p2=None):
    sigma = 10
    a_loss = torch.tensor(0.0,requires_grad=True)

    p1_x, p1_y = p1.exterior.xy 
    p2_x, p2_y = p2.exterior.xy if p2 else [], []

    p1_lenght = len(p1_x)
    p2_lenght = len(p2_x)
    i, j, k = 0,0,3

    while i < p1_lenght-k+1 and j < p2_lenght-k+1:
         #l1 = [(p1_x[i], p1_y[i]),(p1_x[i+1], p1_y[i+1])]
         #l2 = [(p1_x[i+1], p1_y[i+1]), (p1_x[i+2], p1_y[i+2])] 
        p1_ang = getAngle((p1_x[i], p1_y[i]), 
                           (p1_x[i+1], p1_y[i+1]), 
                           (p1_x[i+2], p1_y[i+2])
                          )  #calculate_angle(l1,l2)
         
         #l1 = [(p2_x[j], p2_y[j]),(p2_x[j+1], p2_y[j+1])]
         #l2 = [(p2_x[j+1], p2_y[j+1]), (p2_x[j+2], p2_y[j+2])] 
         
        p2_ang = getAngle( (p2_x[j], p2_y[j]),
                           (p2_x[j+1], p2_y[j+1]),
                           (p2_x[j+2], p2_y[j+2])
                           ) #calculate_angle(l1,l2)
         
        a_loss = a_loss +1- torch.exp(-sigma * abs(p1_ang - p2_ang) )

        i +=1
        j +=1
    while i < p1_lenght-k +1:
        
        #l1 = [(p1_x[i], p1_y[i]),(p1_x[i+1], p1_y[i+1])]
        #l2 = [(p1_x[i+1], p1_y[i+1]), (p1_x[i+2], p1_y[i+2])] 
        
        a_l= getAngle((p1_x[i], p1_y[i]),
                      (p1_x[i+1], p1_y[i+1]),
                      (p1_x[i+2], p1_y[i+2])
                     )
        a_loss= a_loss+ 1- torch.exp(-(sigma *a_l))
        i +=1

    while j < p2_lenght-k +1:
        #l1 = [(p2_x[j], p2_y[j]),(p2_x[j+1], p2_y[j+1])]
        #l2 = [(p2_x[j+1], p2_y[j+1]), (p2_x[j+2], p2_y[j+2])] 
        
        a_l = getAngle((p2_x[j], p2_y[j]),
                      (p2_x[j+1], p2_y[j+1]),
                      (p2_x[j+2], p2_y[j+2]))
        
        a_loss = a_loss+  1 - torch.exp(-(sigma * a_l))
        j +=1
  
    return a_loss
