
## WIP: Come back in a couple of weeks
# RetinaNet - Focal Loss for Dense Object Detection
![architecture](architecture.png)

## Outputs
<a href="tpu/outputs/c5b2506d-9121123c.jpg" target="_blank"><img 
src="tpu/outputs/c5b2506d-9121123c.jpg" alt="not available_1" title="predicted boxes" width="1280" height="720" 
border="10" /></a>
<a href="tpu/outputs/c5b2506d-aa9e5484.jpg " target="_blank"><img 
src="tpu/outputs/c5b2506d-aa9e5484.jpg" alt="not available_2" title="predicted boxes" width="1280" height="720" 
border="10" /></a>



## Progress Tracker
 * [x] Implement data input pipeline with tf.data
 * [x] Implement train and validation steps
 * [x] Add random_flip
 * [x] TPU support
 * [ ] LR schedule and handle large batches (TPU)
 * [x] Train on BDD dataset
 * [x] Add inference code
 * [ ] Publish results
 * [ ] Add support for video inference
 * [ ] Code clean up & refactoring
