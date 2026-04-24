mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/fused_decode_attn.so --shared src/fused_decode_attn.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/paged_decode_attn.so --shared src/paged_decode_attn.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/flash_decode_attn.so --shared src/flash_decode_attn.cu -Xcompiler -fPIC
