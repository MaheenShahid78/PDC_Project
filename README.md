# Parallel Image Processing (MPI + OpenMP)

A hybrid **MPI + OpenMP** image processing system written in C++ that applies filters to PNG and PPM images using two levels of parallelism: distributed processes (MPI) and shared-memory threads (OpenMP).
---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [MPI Functions Used](#mpi-functions-used)
- [OpenMP Clauses Used](#openmp-clauses-used)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Usage](#usage)
- [Filters and Default Parameters](#filters-and-default-parameters)
- [Sample Output](#sample-output)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Team](#team)

---

## Overview

Image processing operations such as filtering, enhancement, and edge detection are computationally expensive on large, high-resolution images. This project speeds them up with a hybrid parallel design:

- **MPI (Message Passing Interface)** distributes blocks of image rows across multiple processes, running on different cores or machines.
- **OpenMP** parallelizes pixel-level work with multi-threading inside each process.

Both **PNG** and **PPM (P6)** formats are supported. PNG inputs are automatically decoded to PPM for processing, and the result can be saved back as PNG.

## Features

- **Hybrid parallel-distributed architecture**: coarse-grained (MPI) plus fine-grained (OpenMP) parallelism
- **Automatic format conversion** between PNG and PPM using [LodePNG](https://lodev.org/lodepng/)
- **8 image operations**: grayscale, Gaussian blur (5x5), Sobel edge detection, binary thresholding, sharpening, brightness, contrast, and color quantization compression
- **Performance monitoring**: per-rank processing time and live progress bars
- **Scalable workload distribution**: handles images of any size, with remainder rows spread across ranks

## Architecture

### Step 1: MPI Distribution (coarse-grained)

1. Rank 0 reads the full image (decoding PNG to PPM if needed).
2. `MPI_Bcast` sends the image dimensions to all processes.
3. `MPI_Scatterv` distributes contiguous row chunks to each process (for example, rows 0-359, 360-719, and so on).

### Step 2: OpenMP Processing (fine-grained)

1. Inside each MPI process, OpenMP threads are spawned.
2. Threads divide that process's rows among themselves.
3. All threads share the process's local image array.
4. The selected filter is applied in parallel.

### Step 3: MPI Collection

1. `MPI_Gatherv` collects the processed chunks back to rank 0.
2. Rank 0 writes the final image to disk (converting back to PNG if requested).

```
                    +------------------+
                    |      Rank 0      |
                    |  read full image |
                    +--------+---------+
                             | MPI_Bcast (dimensions)
                             | MPI_Scatterv (row chunks)
        +--------------------+--------------------+
        v                    v                    v
  +-----------+        +-----------+        +-----------+
  |  Rank 0   |        |  Rank 1   |        |  Rank N-1 |
  | rows 0..k |        | rows k..m |        |    ...    |
  | OpenMP    |        | OpenMP    |        | OpenMP    |
  | threads   |        | threads   |        | threads   |
  +-----+-----+        +-----+-----+        +-----+-----+
        +--------------------+--------------------+
                             | MPI_Gatherv
                             v
                    +------------------+
                    |      Rank 0      |
                    | write final image|
                    +------------------+
```

## MPI Functions Used

| Function | Purpose | Distributed Concept |
|---|---|---|
| `MPI_Init()` | Initializes the MPI environment | Process spawning |
| `MPI_Comm_rank()` | Gets the process ID (0, 1, 2, ...) | Process identification |
| `MPI_Comm_size()` | Gets the total process count | Process discovery |
| `MPI_Bcast()` | Broadcasts data from one process to all | One-to-all communication |
| `MPI_Scatterv()` | Distributes data chunks to processes | Data distribution (domain decomposition) |
| `MPI_Gatherv()` | Collects results from all processes | Result aggregation |
| `MPI_Wtime()` | High-resolution timer | Performance measurement |
| `MPI_Finalize()` | Cleans up the MPI environment | Graceful shutdown |
| `MPI_Abort()` | Terminates all processes | Fault handling |

## OpenMP Clauses Used

| Clause / Function | Where | Purpose | Parallel Concept |
|---|---|---|---|
| `#pragma omp parallel for` | PNG/PPM conversion | Splits loop iterations among threads | Data parallelism |
| `omp_get_max_threads()` | `main()` | Gets available thread count | Resource discovery |
| `schedule(static)` | `grayscale()`, `brightness()`, `contrast()`, `compress()`, `binaryThreshold()` | Pre-assigns equal chunks to threads | Static load balancing |
| `schedule(dynamic)` | `blur()` | Assigns work to threads at runtime | Dynamic load balancing |
| `collapse(2)` | `blur()`, `edgeDetect()`, `sharpen()` | Merges nested loops | Loop fusion |
| `reduction(+:counter)` | `blur()`, `edgeDetect()`, `sharpen()` | Safely combines thread-local counters | Reduction operation |
| `#pragma omp critical` | All filters (progress updates) | Only one thread executes the block | Mutual exclusion |

## Tech Stack

| Category | Technology |
|---|---|
| Language | C++ |
| Distributed library | MPI (MS-MPI) |
| Shared-memory library | OpenMP |
| Image library | LodePNG (PNG encoding/decoding) |
| IDE / compiler | Visual Studio (MSVC) |
| Execution | `mpiexec` |

## Prerequisites

- Windows with **Visual Studio** (MSVC compiler)
- **MS-MPI** runtime and SDK ([download from Microsoft](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi))
- **LodePNG** (`lodepng.cpp` and `lodepng.h`) added to the project
- OpenMP support enabled in the compiler

## Build

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Open `ParallelImageProcessing.slnx` in **Visual Studio**.
3. Make sure `lodepng.cpp` and `lodepng.h` are part of the project.
4. Configure MS-MPI:
   - Add the MS-MPI **include** directory to *Additional Include Directories*.
   - Add the MS-MPI **lib** directory to *Additional Library Directories*.
   - Link against `msmpi.lib`.
5. Enable OpenMP: *Project Properties > C/C++ > Language > Open MP Support > Yes (/openmp)*.
6. Build in **Release** mode for best performance.

## Usage

The program requires **at least 4 OpenMP threads** per process. Set the thread count before running:

```bat
set OMP_NUM_THREADS=4
```

Run with `mpiexec`:

```bat
mpiexec -n <P> ParallelImageProcessing.exe <input> <output> <operation>
```

| Argument | Description |
|---|---|
| `<P>` | Number of MPI processes |
| `<input>` | Input image (`.png` or `.ppm`) |
| `<output>` | Output image (`.png` or `.ppm`) |
| `<operation>` | One of the operations listed below |

### Examples

```bat
:: Grayscale using 4 MPI processes
mpiexec -n 4 ParallelImageProcessing.exe input.png output.png gray

:: Gaussian blur
mpiexec -n 4 ParallelImageProcessing.exe input.png blurred.png blur

:: Sobel edge detection
mpiexec -n 2 ParallelImageProcessing.exe input.ppm edges.png edge
```

If an unknown operation is given, the program prints the list of valid operations and exits.

## Filters and Default Parameters

| Operation | Description | Default Parameter |
|---|---|---|
| `gray` | Convert to grayscale | n/a |
| `blur` | Gaussian blur | 5x5 kernel |
| `edge` | Sobel edge detection | n/a |
| `threshold` | Binary thresholding | Threshold = 128 |
| `sharpen` | Image sharpening | 3x3 sharpening kernel |
| `bright` | Brightness enhancement | Factor = 1.5 (about 50% brighter) |
| `contrast` | Contrast enhancement | Factor = 1.5 (about 50% more contrast) |
| `compress` | Compression by color quantization | 32 levels |

## Performance Monitoring

Each rank reports its own processing time using `MPI_Wtime()`:

```
Rank 0 finished processing in 0.42 seconds
Rank 1 finished processing in 0.40 seconds
```

Every filter also prints a progress bar while it runs. To measure speedup, compare timings across different values of `-n` and different `OMP_NUM_THREADS` settings on the same image.

## Sample Output

Sample results are stored in the `images/` folder.

| Original | Blur | Bright | Sharp |
|:---:|:---:|:---:|:---:|
| ![Original](images/flower.png) | ![Blur](images/blur_ts.png) | ![Bright](images/bright_ts.png) | ![Sharp](images/sharp_ts.png) |

| Contrast | Grayscale | Thresholding | Edge Detection |
|:---:|:---:|:---:|:---:|
| ![Contrast](images/contrast_ts.png) | ![Grayscale](images/gray_ts.png) | ![Thresholding](images/thres.png) | ![Edge](images/ed_ts.png) |

## Project Structure

```
.
├── parallel_image.cpp                    # main program (MPI + OpenMP)
├── lodepng.cpp                           # PNG encode/decode library
├── lodepng.h
├── ParallelImageProcessing.slnx          # Visual Studio solution
├── ParallelImageProcessing.vcxproj       # project settings (MPI + OpenMP)
├── ParallelImageProcessing.vcxproj.filters
├── images/                               # sample input and filter outputs
├── docs/
│   └── PDC_Project_Report.pdf            # full project report
├── .gitignore
└── README.md
```

## Future Work

- **GPU acceleration** with CUDA/OpenCL for faster, real-time filtering
- **More image formats**: JPEG, TIFF, BMP, and RAW
- **Pipeline parallelism**: chaining multiple filters in a continuous multi-stage pipeline
- **Distributed storage**: HDFS or cloud storage for large-scale datasets
- **User interface**: a GUI for image upload, filter selection, and visualization

## Team

| Name | Registration Number |
|---|---|
| Hoor-ul-ain Wajid | hoorwajid3@gmail.com
| Maheen Shahid | maheenshahid0302@gmail.com
| Kinza Mazhar | kinzamazhar40@gmail.com

## Acknowledgements

- [LodePNG](https://lodev.org/lodepng/) by Lode Vandevenne for PNG encoding/decoding
- Microsoft MPI and OpenMP for the parallel runtime
