#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "lodepng.h"
using namespace std;

// ------------------ PNG <-> PPM Conversion ---------------------
bool pngToPpm(const string& pngFile, const string& ppmFile, int& width, int& height, vector<unsigned char>& data) {
    vector<unsigned char> image;
    unsigned w, h;
    unsigned error = lodepng::decode(image, w, h, pngFile);
    if (error) {
        cerr << "PNG decode error: " << lodepng_error_text(error) << "\n";
        return false;
    }

    width = w; height = h;
    data.resize(width * height * 3);
#pragma omp parallel for
    for (int i = 0; i < int(width * height); i++) {
        data[3 * i] = image[4 * i];
        data[3 * i + 1] = image[4 * i + 1];
        data[3 * i + 2] = image[4 * i + 2];
    }
    ofstream f(ppmFile, ios::binary);
    if (!f) return false;
    f << "P6\n" << width << " " << height << "\n255\n";
    f.write((char*)data.data(), data.size());
    return true;
}
bool ppmToPng(const string& ppmFile, const string& pngFile) {
    ifstream f(ppmFile, ios::binary);
    if (!f) return false;

    string magic; f >> magic;
    if (magic != "P6") return false;
    unsigned width, height, maxval;
    f >> width >> height >> maxval;
    f.get();

    vector<unsigned char> data(width * height * 3);
    f.read((char*)data.data(), data.size());

    vector<unsigned char> pngData(width * height * 4);

#pragma omp parallel for
    for (int i = 0; i < int(width * height); i++) {
        pngData[4 * i] = data[3 * i];
        pngData[4 * i + 1] = data[3 * i + 1];
        pngData[4 * i + 2] = data[3 * i + 2];
        pngData[4 * i + 3] = 255;
    }

    unsigned error = lodepng::encode(pngFile, pngData, width, height);
    if (error) {
        cerr << "PNG encode error: " << lodepng_error_text(error) << "\n";
        return false;
    }
    return true;
}

// ------------------ Utility: Read PPM ---------------------
bool readPPM(const string& filename, int& width, int& height, vector<unsigned char>& data) {
    ifstream f(filename, ios::binary);
    if (!f) return false;

    string magic; f >> magic;
    if (magic != "P6") return false;

    f >> width >> height;
    int maxval;
    f >> maxval;
    f.get();
    data.resize(width * height * 3);
    f.read((char*)data.data(), data.size());
    return true;
}

// ------------------ Write PPM ---------------------
bool writePPM(const string& filename, int width, int height, const vector<unsigned char>& data) {
    ofstream f(filename, ios::binary);
    if (!f) return false;
    f << "P6\n" << width << " " << height << "\n255\n";
    f.write((char*)data.data(), data.size());
    return true;
}

// ------------------ Progress Bar ---------------------
void showProgress(int rank, int done, int total) {
    static int lastPercent = -1;
    int percent = (100 * done) / total;

    if (percent != lastPercent) {
        int barWidth = 40;
        int pos = (barWidth * done) / total;

        cout << "Rank " << rank << " [";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] " << percent << "%\r";
        cout.flush();

        if (done == total) {
            cout << endl;
            lastPercent = -1;
        }
        else {
            lastPercent = percent;
        }
    }
}

// ------------------ Filters ---------------------
void grayscale(vector<unsigned char>& img, int width, int height, int rank) {
    int totalPixels = width * height;
    int progressInterval = max(1, totalPixels / 100);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < totalPixels; i++) {
        int idx = i * 3;
        unsigned char gray = (img[idx] + img[idx + 1] + img[idx + 2]) / 3;
        img[idx] = img[idx + 1] = img[idx + 2] = gray;

        if (i % progressInterval == 0) {
#pragma omp critical
            {
                showProgress(rank, i, totalPixels);
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

// Enhanced blur with configurable kernel size (default 5x5 for stronger blur)
void blur(vector<unsigned char>& img, int width, int height, int rank, int kernelSize = 5) {
    vector<unsigned char> copy = img;
    int halfKernel = kernelSize / 2;
    int totalPixels = (width - 2 * halfKernel) * (height - 2 * halfKernel);
    int progressInterval = max(1, totalPixels / 100);
    int counter = 0;
    int kernelArea = kernelSize * kernelSize;

#pragma omp parallel for collapse(2) reduction(+:counter) schedule(dynamic)
    for (int y = halfKernel; y < height - halfKernel; y++) {
        for (int x = halfKernel; x < width - halfKernel; x++) {
            int sumR = 0, sumG = 0, sumB = 0;

            // Apply larger kernel for more blur
            for (int dy = -halfKernel; dy <= halfKernel; dy++) {
                for (int dx = -halfKernel; dx <= halfKernel; dx++) {
                    int idx = ((y + dy) * width + (x + dx)) * 3;
                    sumR += copy[idx];
                    sumG += copy[idx + 1];
                    sumB += copy[idx + 2];
                }
            }

            int idx = (y * width + x) * 3;
            img[idx] = sumR / kernelArea;
            img[idx + 1] = sumG / kernelArea;
            img[idx + 2] = sumB / kernelArea;
            counter++;

            if (counter % progressInterval == 0) {
#pragma omp critical
                {
                    showProgress(rank, counter, totalPixels);
                }
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

void edgeDetect(vector<unsigned char>& img, int width, int height, int rank) {
    vector<unsigned char> copy = img;
    int Gx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
    int Gy[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
    int totalPixels = (width - 2) * (height - 2);
    int progressInterval = max(1, totalPixels / 100);
    int counter = 0;

#pragma omp parallel for collapse(2) reduction(+:counter)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0, sumY = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int idx = ((y + dy) * width + (x + dx)) * 3;
                    unsigned char gray = (copy[idx] + copy[idx + 1] + copy[idx + 2]) / 3;
                    sumX += gray * Gx[dy + 1][dx + 1];
                    sumY += gray * Gy[dy + 1][dx + 1];
                }
            }
            int mag = min(255, static_cast<int>(sqrt(sumX * sumX + sumY * sumY)));
            int idx = (y * width + x) * 3;
            img[idx] = img[idx + 1] = img[idx + 2] = mag;
            counter++;

            if (counter % progressInterval == 0) {
#pragma omp critical
                {
                    showProgress(rank, counter, totalPixels);
                }
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

// NEW: Binary Thresholding (threshold = 128 by default)
void binaryThreshold(vector<unsigned char>& img, int width, int height, int rank, int threshold = 128) {
    int totalPixels = width * height;
    int progressInterval = max(1, totalPixels / 100);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < totalPixels; i++) {
        int idx = i * 3;
        unsigned char gray = (img[idx] + img[idx + 1] + img[idx + 2]) / 3;
        unsigned char binary = (gray >= threshold) ? 255 : 0;
        img[idx] = img[idx + 1] = img[idx + 2] = binary;

        if (i % progressInterval == 0) {
#pragma omp critical
            {
                showProgress(rank, i, totalPixels);
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

// NEW: Sharpening filter
void sharpen(vector<unsigned char>& img, int width, int height, int rank) {
    vector<unsigned char> copy = img;
    // Sharpening kernel
    int kernel[3][3] = { {0,-1,0},{-1,5,-1},{0,-1,0} };
    int totalPixels = (width - 2) * (height - 2);
    int progressInterval = max(1, totalPixels / 100);
    int counter = 0;

#pragma omp parallel for collapse(2) reduction(+:counter)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumR = 0, sumG = 0, sumB = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int idx = ((y + dy) * width + (x + dx)) * 3;
                    int k = kernel[dy + 1][dx + 1];
                    sumR += copy[idx] * k;
                    sumG += copy[idx + 1] * k;
                    sumB += copy[idx + 2] * k;
                }
            }

            int idx = (y * width + x) * 3;
            img[idx] = min(255, max(0, sumR));
            img[idx + 1] = min(255, max(0, sumG));
            img[idx + 2] = min(255, max(0, sumB));
            counter++;

            if (counter % progressInterval == 0) {
#pragma omp critical
                {
                    showProgress(rank, counter, totalPixels);
                }
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

// NEW: Brightness Enhancement (factor: 1.5 = 50% brighter)
void brightness(vector<unsigned char>& img, int width, int height, int rank, float factor = 1.5) {
    int totalPixels = width * height;
    int progressInterval = max(1, totalPixels / 100);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < totalPixels; i++) {
        int idx = i * 3;
        img[idx] = min(255, static_cast<int>(img[idx] * factor));
        img[idx + 1] = min(255, static_cast<int>(img[idx + 1] * factor));
        img[idx + 2] = min(255, static_cast<int>(img[idx + 2] * factor));

        if (i % progressInterval == 0) {
#pragma omp critical
            {
                showProgress(rank, i, totalPixels);
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

// NEW: Contrast Enhancement (factor: 1.5 = 50% more contrast)
void contrast(vector<unsigned char>& img, int width, int height, int rank, float factor = 1.5) {
    int totalPixels = width * height;
    int progressInterval = max(1, totalPixels / 100);
    const float midpoint = 128.0f;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < totalPixels; i++) {
        int idx = i * 3;

        for (int c = 0; c < 3; c++) {
            float pixel = img[idx + c];
            float adjusted = ((pixel - midpoint) * factor) + midpoint;
            img[idx + c] = min(255, max(0, static_cast<int>(adjusted)));
        }

        if (i % progressInterval == 0) {
#pragma omp critical
            {
                showProgress(rank, i, totalPixels);
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

// NEW: Image Compression (quality reduction by downsampling color depth)
void compress(vector<unsigned char>& img, int width, int height, int rank, int levels = 32) {
    int totalPixels = width * height;
    int progressInterval = max(1, totalPixels / 100);
    int divisor = 256 / levels;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < totalPixels; i++) {
        int idx = i * 3;
        img[idx] = (img[idx] / divisor) * divisor;
        img[idx + 1] = (img[idx + 1] / divisor) * divisor;
        img[idx + 2] = (img[idx + 2] / divisor) * divisor;

        if (i % progressInterval == 0) {
#pragma omp critical
            {
                showProgress(rank, i, totalPixels);
            }
        }
    }
    showProgress(rank, totalPixels, totalPixels);
}

// ------------------ MAIN: MPI + OpenMP ---------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check minimum thread requirement (4 threads minimum)
    int numThreads = omp_get_max_threads();
    if (numThreads < 4) {
        if (rank == 0) {
            cerr << "Error: Minimum 4 OpenMP threads required. Current: " << numThreads << "\n";
            cerr << "Set OMP_NUM_THREADS environment variable to 4 or more.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        cout << "Running with " << size << " MPI processes and " << numThreads << " OpenMP threads per process\n";
    }

    if (argc < 4) {
        if (rank == 0) {
            cout << "Usage: mpiexec -n <P> " << argv[0] << " input.{ppm|png} output.{ppm|png} [operation]\n";
            cout << "Operations:\n";
            cout << "  gray         - Convert to grayscale\n";
            cout << "  blur         - Apply 5x5 Gaussian blur (stronger)\n";
            cout << "  edge         - Sobel edge detection\n";
            cout << "  threshold    - Binary thresholding (128)\n";
            cout << "  sharpen      - Sharpen image\n";
            cout << "  bright       - Increase brightness (1.5x)\n";
            cout << "  contrast     - Enhance contrast (1.5x)\n";
            cout << "  compress     - Compress by color quantization (32 levels)\n";
        }
        MPI_Finalize();
        return 0;
    }

    string input = argv[1], output = argv[2], op = argv[3];
    int width, height;
    vector<unsigned char> fullImg;

    bool isPngInput = input.substr(input.find_last_of('.') + 1) == "png";
    string tmpPPM = "tmp_input.ppm";

    if (rank == 0) {
        if (isPngInput) {
            if (!pngToPpm(input, tmpPPM, width, height, fullImg)) {
                cerr << "Error: Failed to convert PNG to PPM\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else {
            if (!readPPM(input, width, height, fullImg)) {
                cerr << "Error: Failed to read PPM file\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rowsPerProc = height / size;
    int remainder = height % size;
    vector<int> sendCounts(size), displs(size);
    int offset = 0;

    for (int i = 0; i < size; i++) {
        int rows = rowsPerProc + (i < remainder ? 1 : 0);
        sendCounts[i] = rows * width * 3;
        displs[i] = offset;
        offset += sendCounts[i];
    }

    int localCount = sendCounts[rank];
    vector<unsigned char> localImg(localCount);

    MPI_Scatterv(fullImg.data(), sendCounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
        localImg.data(), localCount, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int localRows = localCount / (width * 3);
    double t_start = MPI_Wtime();

    // Apply selected filter
    if (op == "gray") {
        grayscale(localImg, width, localRows, rank);
    }
    else if (op == "blur") {
        blur(localImg, width, localRows, rank, 5); // 5x5 kernel for stronger blur
    }
    else if (op == "edge") {
        edgeDetect(localImg, width, localRows, rank);
    }
    else if (op == "threshold") {
        binaryThreshold(localImg, width, localRows, rank, 128);
    }
    else if (op == "sharpen") {
        sharpen(localImg, width, localRows, rank);
    }
    else if (op == "bright") {
        brightness(localImg, width, localRows, rank, 1.5);
    }
    else if (op == "contrast") {
        contrast(localImg, width, localRows, rank, 1.5);
    }
    else if (op == "compress") {
        compress(localImg, width, localRows, rank, 32);
    }
    else {
        if (rank == 0) {
            cerr << "Error: Unknown operation '" << op << "'.\n";
            cerr << "Use: gray, blur, edge, threshold, sharpen, bright, contrast, or compress\n";
        }
    }

    double t_end = MPI_Wtime();
    cout << "Rank " << rank << " finished processing in " << (t_end - t_start) << " seconds\n";

    if (rank == 0) fullImg.assign(width * height * 3, 0);

    MPI_Gatherv(localImg.data(), localCount, MPI_UNSIGNED_CHAR,
        fullImg.data(), sendCounts.data(), displs.data(),
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        bool isPngOutput = output.substr(output.find_last_of('.') + 1) == "png";
        string finalPPM = isPngOutput ? "tmp_output.ppm" : output;

        if (!writePPM(finalPPM, width, height, fullImg)) {
            cerr << "Error: Failed to write PPM file\n";
            MPI_Finalize();
            return 1;
        }

        if (isPngOutput) {
            if (!ppmToPng(finalPPM, output)) {
                cerr << "Error: Failed to convert PPM to PNG\n";
                MPI_Finalize();
                return 1;
            }
        }

        cout << "Done. Output saved to " << output << "\n";
    }

    MPI_Finalize();
    return 0;
}