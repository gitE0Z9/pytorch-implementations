from object_detection.models.yolov2.anchor import PriorBox

prior_box = PriorBox(5, "voc")

anchors = prior_box.build_anchors()

print(prior_box.anchors.shape)
