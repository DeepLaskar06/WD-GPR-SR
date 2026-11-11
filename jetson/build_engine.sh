#!/bin/bash
# Execute on NVIDIA Jetson Orin Nano
/usr/src/tensorrt/bin/trtexec --onnx=models\deep_branch.onnx --saveEngine=models\deep_branch.engine --fp16
