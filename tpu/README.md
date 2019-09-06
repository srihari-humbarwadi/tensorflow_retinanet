  ## Some things to note
  - Currently there is no proper support for TPU on tensorflow 2.0rc0.
  - The training under this directory only work for tensorflow 1.14.0.
  - The tfrecords should be uploaded to GCS bucket since TPU does not support local file system.
  - Each epoch takes around 10 mins on google colaboratory with TPUv2-8 runtime.
