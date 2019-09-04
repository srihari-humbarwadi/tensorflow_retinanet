# RetinaNet - Focal Loss for Dense Object Detection
![architecture](architecture.png)


<a href="test_box_encoding.png " target="_blank"><img 
src="test_box_encoding.png" alt="test_box_encoding" title="ground truth boxes" width="430" height="215" 
border="10" /></a>
<a href="test_anchor_matching.png " target="_blank"><img 
src="test_anchor_matching.png" alt="test_anchor_matching" title="matched anchor boxes" width="430" height="215" 
border="10" /></a>

## Progress Tracker
 - [x] Implement data input pipeline with tf.data
 - [x] Implement train and validation steps
 - [x] Add random_flip
 - [x] TPU support
 - [ ] LR schedule and handle large batches (TPU)
 - [ ] Train on BDD dataset
 - [x] Add inference code
 - [ ] Publish results
 - [ ] Add support for video inference
 - [ ] Code clean up & refactoring
