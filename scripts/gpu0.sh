# Source before every run: confine CUDA (torch and brotensor) and bro's Vulkan
# device to GPU 0, the 4090 on PCI bus 01:00.0. GPU 1 (bus 02:00.0) drives the
# desktop and must stay free. BRO_VK_DEVICE=1 is the Vulkan index that lands
# on bus 01 on this machine (Vulkan's order is not nvidia-smi's). Check with
#   nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv
# that nothing of ours is on bus 02 before a long run.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export BRO_VK_DEVICE=1
export PYTHONUTF8=1
