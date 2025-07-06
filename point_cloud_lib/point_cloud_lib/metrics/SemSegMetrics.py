import numpy as np

class SemSegMetrics:

    def __init__(self, p_num_classes, p_mask_classes):
        self.num_classes_ = p_num_classes
        mask = []
        for i in range(self.num_classes_):
            if i in p_mask_classes:
                mask.append(False)
            else:
                mask.append(True)
        self.mask_ = np.array(mask)
        self.accum_intersection_ = np.array([0.0 for i in range(self.num_classes_)])
        self.accum_union_ = np.array([0.0 for i in range(self.num_classes_)])
        self.accum_gt_ = np.array([0.0 for i in range(self.num_classes_)])


    def reset(self):
        self.accum_intersection_ = np.array([0.0 for i in range(self.num_classes_)])
        self.accum_union_ = np.array([0.0 for i in range(self.num_classes_)])
        self.accum_gt_ = np.array([0.0 for i in range(self.num_classes_)])
        

    def update_metrics(self, p_predict_probs, p_labels):

        pred = np.argmax(p_predict_probs, 1)
        mask_eq = pred == p_labels
        
        num_labels = np.bincount(p_labels, 
            minlength=self.num_classes_).astype(np.float32)
        num_pred = np.bincount(pred, 
            minlength=self.num_classes_).astype(np.float32)
        num_equal = np.bincount(p_labels[mask_eq], 
            minlength=self.num_classes_).astype(np.float32)

        self.accum_gt_ += num_labels
        self.accum_union_ += num_labels + num_pred - num_equal
        self.accum_intersection_ += num_equal

    
    def per_class_acc(self):
        class_acc = self.accum_intersection_[self.mask_]/np.maximum(self.accum_gt_[self.mask_], 1)        
        return class_acc*100.0


    def per_class_iou(self):
        class_iou = self.accum_intersection_[self.mask_]/np.maximum(self.accum_union_[self.mask_], 1)
        return class_iou*100.0
    
    def class_mean_acc(self):
        class_acc = self.accum_intersection_[self.mask_]/np.maximum(self.accum_gt_[self.mask_], 1)        
        return np.mean(class_acc)*100.0


    def class_mean_iou(self):
        class_iou = self.accum_intersection_[self.mask_]/np.maximum(self.accum_union_[self.mask_], 1)
        return np.mean(class_iou)*100.0

    
    def mean_acc(self):
        result = np.sum(self.accum_intersection_[self.mask_])/np.maximum(np.sum(self.accum_gt_[self.mask_]), 1)        
        return result*100.0


    def mean_iou(self):
        result = np.sum(self.accum_intersection_[self.mask_])/np.maximum(np.sum(self.accum_union_[self.mask_]), 1)
        return result*100.0

