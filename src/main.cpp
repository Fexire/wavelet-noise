#include <math.h>
#include <iostream>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* Note: this code is designed for brevity, not efficiency; many operations can be hoisted,
 * precomputed, or vectorized. Some of the straightforward details, such as tile meshing,
 * decorrelating bands and fading out the last band, are omitted in the interest of space.*/
static float *noiseTileData;
static int noiseTileSize;

std::normal_distribution<float> gaussianNoise(0, 1);
std::default_random_engine generator;

int Mod(int x, int n)
{
    int m = x % n;
    return (m < 0) ? m + n : m;
}
#define ARAD 16
void Downsample(float *from, float *to, int n, int stride)
{
    float *a, aCoeffs[2 * ARAD] = {
                  0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002172, 0.019120,
                  -0.005040, -0.044412, 0.011655, 0.103311, -0.025936, -0.243780, 0.033979, 0.655340,
                  0.655340, 0.033979, -0.243780, -0.025936, 0.103311, 0.011655, -0.044412, -0.005040,
                  0.019120, 0.002172, -0.008233, -0.000938, 0.003546, 0.000410, -0.001528, 0.000334};
    a = &aCoeffs[ARAD];
    for (int i = 0; i < n / 2; i++)
    {
        to[i * stride] = 0;
        for (int k = 2 * i - ARAD; k < 2 * i + ARAD; k++)
        {
            to[i * stride] += a[k - 2 * i] * from[Mod(k, n) * stride];
        }
    }
}
void Upsample(float *from, float *to, int n, int stride)
{
    float *p, pCoeffs[4] = {0.25, 0.75, 0.75, 0.25};
    p = &pCoeffs[2];
    for (int i = 0; i < n; i++)
    {
        to[i * stride] = 0;
        for (int k = i / 2; k <= i / 2 + 1; k++)
        {
            to[i * stride] += p[i - 2 * k] * from[Mod(k, n / 2) * stride];
        }
    }
}

void GenerateNoiseTile(int n)
{
    if (n % 2)
        n++; /* tile size must be even */
    int ix, iy, iz, i, sz = n * n * n * sizeof(float);
    float *temp1 = (float *)malloc(sz), *temp2 = (float *)malloc(sz), *noise = (float *)malloc(sz);
    /* Step 1. Fill the tile with random numbers in the range -1 to 1. */
    for (i = 0; i < n * n * n; i++)
    {
        noise[i] = gaussianNoise(generator) / 4.;
    }
    /* Steps 2 and 3. Downsample and upsample the tile */
    for (iy = 0; iy < n; iy++)
    {
        for (iz = 0; iz < n; iz++)
        { /* each x row */
            i = iy * n + iz * n * n;
            Downsample(&noise[i], &temp1[i], n, 1);
            Upsample(&temp1[i], &temp2[i], n, 1);
        }
    }
    for (ix = 0; ix < n; ix++)
    {
        for (iz = 0; iz < n; iz++)
        { /* each y row */
            i = ix + iz * n * n;
            Downsample(&temp2[i], &temp1[i], n, n);
            Upsample(&temp1[i], &temp2[i], n, n);
        }
    }
    for (ix = 0; ix < n; ix++)
    {
        for (iy = 0; iy < n; iy++)
        { /* each z row */
            i = ix + iy * n;
            Downsample(&temp2[i], &temp1[i], n, n * n);
            Upsample(&temp1[i], &temp2[i], n, n * n);
        }
    }
    /* Step 4. Subtract out the coarse-scale contributionnoiseTileData */
    for (i = 0; i < n * n * n; i++)
    {
        noise[i] -= temp2[i];
    }
    /* Avoid even/odd variance difference by adding odd-offset version of noise to itself.*/
    int offset = n / 2;
    if (offset % 2 == 0)
        offset++;
    for (i = 0, ix = 0; ix < n; ix++)
        for (iy = 0; iy < n; iy++)
            for (iz = 0; iz < n; iz++)
                temp1[i++] = noise[Mod(ix + offset, n) + Mod(iy + offset, n) * n + Mod(iz + offset, n) * n * n];
    for (i = 0; i < n * n * n; i++)
    {
        noise[i] += temp1[i];
    }

    noiseTileData = noise;
    noiseTileSize = n;
    free(temp1);
    free(temp2);
}

float WNoise(float p[3])
{                                                 /* Non-projected 3D noise */
    int i, f[3], c[3], mid[3], n = noiseTileSize; /* f, c = filter, noise coeff indices */
    float w[3][3], t, result = 0;
    /* Evaluate quadratic B-spline basis functions */
    for (i = 0; i < 3; i++)
    {
        mid[i] = ceil(p[i] - 0.5);
        t = mid[i] - (p[i] - 0.5);
        w[i][0] = t * t / 2;
        w[i][2] = (1 - t) * (1 - t) / 2;
        w[i][1] = 1 - w[i][0] - w[i][2];
    }
    /* Evaluate noise by weighting noise coefficients by basis function values */
    for (f[2] = -1; f[2] <= 1; f[2]++)
        for (f[1] = -1; f[1] <= 1; f[1]++)
            for (f[0] = -1; f[0] <= 1; f[0]++)
            {
                float weight = 1;
                for (i = 0; i < 3; i++)
                {
                    c[i] = Mod(mid[i] + f[i], n);
                    weight *= w[i][f[i] + 1];
                }
                result += weight * noiseTileData[c[2] * n * n + c[1] * n + c[0]];
            }
    return result;
}
float WProjectedNoise(float p[3], float normal[3])
{                                                   /* 3D noise projected onto 2D */
    int i, c[3], min[3], max[3], n = noiseTileSize; /* c = noise coeff location */
    float result = 0;
    /* Bound the support of the basis functions for this projection direction */
    for (i = 0; i < 3; i++)
    {
        // support = 3 * abs(normal[i]) + 3 * sqrt((1 - normal[i] * normal[i]) / 2);
        min[i] = ceil(p[i] - (3 * abs(normal[i]) + 3 * sqrt((1 - normal[i] * normal[i]) / 2)));
        max[i] = floor(p[i] + (3 * abs(normal[i]) + 3 * sqrt((1 - normal[i] * normal[i]) / 2)));
    }
    /* Loop over the noise coefficients within the bound. */
    for (c[2] = min[2]; c[2] <= max[2]; c[2]++)
    {
        for (c[1] = min[1]; c[1] <= max[1]; c[1]++)
        {
            for (c[0] = min[0]; c[0] <= max[0]; c[0]++)
            {
                float t, t1, t2, t3, dot = 0, weight = 1;
                /* Dot the normal with the vector from c to p */
                for (i = 0; i < 3; i++)
                {
                    dot += normal[i] * (p[i] - c[i]);
                }
                /* Evaluate the basis function at c moved halfway to p along the normal. */
                for (i = 0; i < 3; i++)
                {
                    t = (c[i] + normal[i] * dot / 2) - (p[i] - 1.5);
                    t1 = t - 1;
                    t2 = 2 - t;
                    t3 = 3 - t;
                    weight *= (t <= 0 || t >= 3) ? 0 : (t < 1) ? t * t / 2
                                                   : (t < 2)   ? 1 - (t1 * t1 + t2 * t2) / 2
                                                               : t3 * t3 / 2;
                }
                /* Evaluate noise by weighting noise coefficients by basis function values. */
                result += weight * noiseTileData[Mod(c[2], n) * n * n + Mod(c[1], n) * n + Mod(c[0], n)];
            }
        }
    }
    return result;
}
float WMultibandNoise(float p[3], float s, float *normal, int firstBand, int nbands, float *w)
{
    float q[3], result = 0, variance = 0;
    int i, b;
    for (b = 0; b < nbands && s + firstBand + b < 0; b++)
    {
        for (i = 0; i <= 2; i++)
        {
            q[i] = 2 * p[i] * pow(2, firstBand + b);
        }
        result += (normal) ? w[b] * WProjectedNoise(q, normal) : w[b] * WNoise(q);
    }
    for (b = 0; b < nbands; b++)
    {
        variance += w[b] * w[b];
    }
    /* Adjust the noise so it has a variance of 1. */
    if (variance)
        result /= sqrt(variance * ((normal) ? 0.296 : 0.210));
    return result;
}

cv::Mat nonProjected3Dnoise(float scale, int size, int nbands, float *w)
{
    cv::Mat noiseImage = cv::Mat(size, size, CV_32F);
    float p[3] = {0, 0, 0};
    for (p[0] = 0; p[0] < size; p[0]++)
    {
        for (p[1] = 0; p[1] < size; p[1]++)
        {
            noiseImage.at<float>(p[0], p[1]) = (WMultibandNoise(p, scale, nullptr, -nbands, nbands, w) + 1.) / 2.;
        }
    }
    return noiseImage;
}

cv::Mat projected3Dnoise(float scale, int size, int nbands, float *w, float *normals)
{
    cv::Mat noiseImage = cv::Mat(size, size, CV_32F);
    float p[3] = {0, 0, 0};
    for (p[0] = 0; p[0] < size; p[0]++)
    {
        for (p[1] = 0; p[1] < size; p[1]++)
        {
            noiseImage.at<float>(p[0], p[1]) = (WMultibandNoise(p, scale, normals, -nbands, nbands, w) + 1.) / 2.;
        }
    }
    return noiseImage;
}

cv::Mat dft(cv::Mat &src)
{
    cv::Mat padded; // expand input image to optimal size
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols); // on the border add zero values

    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    merge(planes, 2, complexI); // Add to the expanded another plane with zeros
    dft(complexI, complexI);    // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];
    magI += cv::Scalar::all(1); // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                                // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                                  // viewable image form (float between values 0 and 1).
    return magI;
}

void constantDistribution(int nbands, float *w)
{
    for (int i = 0; i < nbands; i++)
    {
        w[i] = 1;
    }
}

void gaussianDistribution(int nbands, float *w)
{
    float sigma = 1;
    float max = 0;
    for (int i = -nbands / 2; i < nbands / 2; i++)
    {
        w[i + nbands / 2] = (1 / sigma * sqrt(2 * M_PI)) * exp(-0.5 * pow(i / sigma, 2));
        if (w[i + nbands / 2] > max)
        {
            max = w[i + nbands / 2];
        }
    }
    for (int i = 0; i < nbands; i++)
    {
        w[i] /= max;
    }
}

void cosDistribution(int nbands, float *w)
{
    float max = 0;
    for (int i = -nbands / 2; i < nbands / 2; i++)
    {
        w[i + nbands / 2] = cos(i);
        if (w[i + nbands / 2] > max)
        {
            max = w[i + nbands / 2];
        }
    }
    for (int i = 0; i < nbands; i++)
    {
        w[i] /= max;
    }
}

void expDistribution(int nbands, float *w)
{
    float max = 0;
    for (int i = -nbands / 2; i < nbands / 2; i++)
    {
        w[i + nbands / 2] =  exp( 2 + i/double(1+i*i));
        if (w[i + nbands / 2] > max)
        {
            max = w[i + nbands / 2];
        }
    }
    for (int i = 0; i < nbands; i++)
    {
        w[i] /= max;
    }
}

int main(int argc, char const *argv[])
{
    int tileSize = 50;
    GenerateNoiseTile(tileSize);
    const int nbands = 4;
    int size = 150;
    float scale = 1;
    float w[nbands];
    gaussianDistribution(nbands, w);
    float *normals = (float *)malloc(size * size * 3 * sizeof(float));
    for (int i = 0; i < size * size * 3; i += 3)
    {
        normals[i] = 0;
        normals[i + 1] = 0;
        normals[i + 2] = 1;
    }
    cv::Mat nonProjectedImage = nonProjected3Dnoise(scale, size, nbands, w);
    cv::Mat projectedImage = projected3Dnoise(scale, size, nbands, w, normals);
    free(normals);
    cv::imshow("Non-Projected 3D noise", nonProjectedImage);
    cv::imshow("DFT Non-Projected 3D noise", dft(nonProjectedImage));
    cv::imshow("Projected 3D noise", projectedImage);
    cv::imshow("DFT Projected 3D noise", dft(projectedImage));

    cv::waitKey(0);
    return 0;
}