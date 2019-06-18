# FPGA_CPU_HARP2

## Overview
Our goal is to explore OpenCL on a CPU-FPGA Heterogeneous Architecture Research Platform (HARP). These applications are designed to use the latest features of heterogeneous architectures such as Shared Virtual Memory (SVM).



## Usage
Clone the repo and copy to your own directories in vLab:
```
git clone https://github.com/jjiantong/fpga_cpu_harp2.git
scp -r FPGA_CPU_HARP2 $USER@ssh-iam.intel-research.net:/homes/$USER/
```

Connect to the vLab access node `ssh-iam.intel-research.net`:
```
ssh $USER@ssh-iam.intel-research.net
```

We support two FPGA classes: `fpga-bdx-opencl` and `fpga-pac-a10`.

### 1. For `fpga-bdx-opencl` class

*The source code for `fpga-bdx-opencl` is in the `bdx-opencl` directory.*


Configure an OpenCL environment:
```
source /export/fpga/bin/setup-fpga-env fpga-bdx-opencl
qsub-fpga
```

Compile the host program:
```
cd fpga_cpu_harp2/bdx-opencl/$APP/
make
```

Compile the OpenCL kernel:
```
cd device
qsub-aoc $NAME.cl
```

You can check the status by running the command `qstat`. When the job complete, you can find a directory $NAME, files $NAME.aoco, $NAME.aocx, and the related log files.

Move $NAME.aocx to the `bin` directory.


Execute the compiled design:
```
cd fpga_cpu_harp2/bdx-opencl/$APP
./bin/$EXEC
```


### 2. For `fpga-pac-a10` class

*The source code for `fpga-pac-a10` is in the `pac-a10` directory.*


Configure an OpenCL environment:
```
source /export/fpga/bin/setup-fpga-env fpga-pac-a10
qsub-fpga
```

Compile the host program:
```
cd fpga_cpu_harp2/pac-a10/$APP/
make
```

Compile the OpenCL kernel:
```
cd device
qsub-aoc $NAME.cl
```

You can check the status by running the command `qstat`. When the job complete, you can find a directory $NAME, files $NAME.aoco, $NAME.aocx, and the related log files.

Move $NAME.aocx to the `bin` directory.


Execute the compiled design:
```
cd fpga_cpu_harp2/pac-a10/$APP
./bin/$EXEC
```


## Notes
1. Atomics not supported in systems with multiported memories. Therefore, for RSCD application, we have to use the near-optimal execution model (SWI+C), instead of the optimal execution model (NDR+C).
2. Due to atomic operations, SVM cannot be used to share buffers between the CPU and the FPGA for some variables in the RSCD application.
