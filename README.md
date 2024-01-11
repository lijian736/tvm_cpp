## A TVM C++ examples set


### Step 1. download TVM source
```
bash scripts/download_tvm.sh
```

### Step 2. build docker file
```
cd dockerfiles
bash build_docker.sh
```

### Step 3. build TVM .so libs
 1. enter the docker container
 2. edit the third_party/tvm/cmake/config.cmake, change `set(USE_LLVM OFF)` to `set(USE_LLVM ON)`
 3. build TVM so libs  
   run the following scripts. if the build successfully, libtvm.so and libtvm_runtime.so will be generated.

    ```
    cd third_party/tvm
    mkdir build
    cp cmake/config.cmake build
    cd build
    cmake .. && make -j
    ```

### Step 4. build the examples
 ```
 mkdir build
 cd build
 cmake .. && make -j
 ```

