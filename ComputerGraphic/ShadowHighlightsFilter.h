#pragma once

#include <opencv.hpp>
#include <vector>
#include "ColorConverter.h"
#include "LabImageProcessor.h"
#include <iostream>

/// <summary>
/// A filter for correcting shadows and highlights of an image.
///
/// Implements a professional algorithm similar to Adobe Photoshop
/// with separate adjustment of shadows and lights, smooth transitions
/// and maintaining color balance.
/// </summary>
class ShadowHighlightsFilter
{
	/// <summary>Shadow lightening power (0.0 - 1.0)</summary>
	float shadowAmount;

    /// <summary>Highlight darkening power (0.0 - 1.0)</summary>
	float highlightAmount;

    /// <summary>Tonal  width (0.0 - 1.0)</summary>
    float tonalWidth;

    /// <summary>The radius of blurring masks (0.0 - 50.0)</summary>
    float blurRadius;

public:

    /// <summary>
    /// Creates a filter with specified parameters
    /// </summary>
    /// <param name="shadows">Shadow lightening power (0.0 - 1.0)</param>
    /// <param name="highlights">Highlight darkening power (0.0 - 1.0)</param>
    /// <param name="width">Tonal width (0.0 - 1.0)</param>
    /// <param name="radius">Blur radius for masks (0.0 - 50.0)</param>
    ShadowHighlightsFilter(float shadows = 0.3f, float highlights = 0.3f, float width = 0.5f, float radius = 15.0f)
        :   shadowAmount(std::max(0.0f, std::min(1.0f, shadows))), 
            highlightAmount(std::max(0.0f, std::min(1.0f, highlights))), 
            tonalWidth(std::max(0.0f, std::min(1.0f, width))), 
            blurRadius(std::max(0.0f, std::min(50.0f, radius)))
    {
    }

    /// <summary>
    /// Applies the filter to an image
    /// </summary>
    /// <param name="inputImage">Input BGR image</param>
    /// <returns>Processed image</returns>
    /// <exception cref="std::invalid_argument">If input image is empty</exception>
    cv::Mat apply(const cv::Mat& inputImage) 
    {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is empty");

        cv::Mat labImage = ColorConverter::BGR2Lab(inputImage);

        std::vector<cv::Mat> labChannels = LabImageProcessor::splitLab(labImage);
        cv::Mat luminance = labChannels[0];

        cv::Mat luminanceFloat = normalizeLuminance(luminance);

        cv::Mat shadowMask = createAdvancedShadowMask(luminanceFloat);
        cv::Mat highlightMask = createAdvancedHighlightMask(luminanceFloat);

        cv::Mat correctedLuminance = applyAdvancedCorrection(luminanceFloat, shadowMask, highlightMask);

        return convertBackToBGR(correctedLuminance, labChannels);
    }

    /// <summary>
    /// Sets the shadow lightening power
    /// </summary>
    /// <param name="amount">Value from 0.0 to 1.0</param>
    void setShadowAmount(float amount) 
    { 
        shadowAmount = std::max(0.0f, std::min(1.0f, amount)); 
    }

    /// <summary>
    /// Sets the highlight darkening power
    /// </summary>
    /// <param name="amount">Value from 0.0 to 1.0</param>
    void setHighlightAmount(float amount) 
    { 
        highlightAmount = std::max(0.0f, std::min(1.0f, amount)); 
    }

    /// <summary>
    /// Sets the tonal width
    /// </summary>
    /// <param name="width">Value from 0.0 to 1.0</param>
    void setTonalWidth(float width) 
    { 
        tonalWidth = std::max(0.0f, std::min(1.0f, width)); 
    }

    /// <summary>
    /// Sets the blur radius for masks
    /// </summary>
    /// <param name="radius">Value from 0.0 to 50.0</param>
    void setBlurRadius(float radius) 
    { 
        blurRadius = std::max(0.0f, std::min(50.0f, radius)); 
    }

    /// <summary>
    /// Prints current filter settings
    /// </summary>
    void printCurrentSettings() const 
    {
        std::cout << "Current Shadow/Highlights parameters:\nShadow Amount: " << shadowAmount * 100 << "%\nHighlight Amount: " << highlightAmount * 100 << "%\nTonal Width: " << tonalWidth << "\nBlur Radius: " << blurRadius << " px";
    }
private:

    /// <summary>
    /// Normalizes luminance values to range 0-1
    /// </summary>
    /// <param name="luminance">Lab channel luminance matrix</param>
    /// <returns>Normalized luminance matrix</returns>
	cv::Mat normalizeLuminance(const cv::Mat& luminance) const
	{
		cv::Mat result(luminance.size(), CV_32F);

		for (int y = 0; y < luminance.rows; y++)
			for (int x = 0; x < luminance.cols; x++) 
			{
				float value = luminance.at<float>(y, x) / 255.0f;
				result.at<float>(y, x) = value;
			}

		return result;
	}

    /// <summary>
    /// Denormalizes luminance values to range 0-255
    /// </summary>
    /// <param name="normalizedLuminance">Normalized luminance matrix</param>
    /// <returns>Denormalized luminance matrix</returns>
	cv::Mat denormalizeLuminance(const cv::Mat& normalizedLuminance) 
	{
		cv::Mat result(normalizedLuminance.size(), CV_32F);

		for (int y = 0; y < normalizedLuminance.rows; y++)
			for (int x = 0; x < normalizedLuminance.cols; x++) 
			{
				float value = normalizedLuminance.at<float>(y, x) * 255.0f;
				result.at<float>(y, x) = value;
			}

		return result;
	}

    /// <summary>
    /// Applies advanced luminance correction using masks
    /// </summary>
    /// <param name="luminance">Luminance matrix</param>
    /// <param name="shadowMask">Shadow mask</param>
    /// <param name="highlightMask">Highlight mask</param>
    /// <returns>Corrected luminance matrix</returns>
    cv::Mat applyAdvancedCorrection(const cv::Mat& luminance, const cv::Mat& shadowMask, const cv::Mat& highlightMask) const
    {
        cv::Mat result = luminance.clone();

        for (int y = 0; y < result.rows; y++)
            for (int x = 0; x < result.cols; x++)
            {
                float lum = luminance.at<float>(y, x), shadow = shadowMask.at<float>(y, x), highlight = highlightMask.at<float>(y, x);

                float shadowCorrection = shadowAmount * shadow * (1.0f - lum) * 0.3f, highlightCorrection = highlightAmount * highlight * lum * 0.3f;

                float corrected = lum + shadowCorrection - highlightCorrection;

                float minVal = lum * 0.5f, maxVal = 1.0f - (1.0f - lum) * 0.5f;

                result.at<float>(y, x) = std::max(minVal, std::min(maxVal, corrected));
            }

        return result;
    }

    /// <summary>
    /// Converts back to BGR color space
    /// </summary>
    /// <param name="correctedLuminance">Corrected luminance</param>
    /// <param name="labChannels">Lab image channels</param>
    /// <returns>BGR image</returns>
	cv::Mat convertBackToBGR(const cv::Mat& correctedLuminance, std::vector<cv::Mat>& labChannels) 
	{
		cv::Mat denormalizedLuminance = denormalizeLuminance(correctedLuminance);
		labChannels[0] = denormalizedLuminance;
		cv::Mat resultLab = LabImageProcessor::mergeLab(labChannels);
		return ColorConverter::Lab2BGR(resultLab);
	}

    /// <summary>
    /// Applies Gaussian blur to an image
    /// </summary>
    /// <param name="input">Input image</param>
    /// <param name="radius">Blur radius</param>
    /// <returns>Blurred image</returns>
    cv::Mat applyGaussianBlur(const cv::Mat& input, float radius) const 
    {
        if (radius < 0.1f) 
            return input.clone();

        cv::Mat result = input.clone();
        int kernelSize = std::max(3, (int)(radius * 2 + 1) | 1);

        cv::Mat kernel = createGaussianKernel(kernelSize, radius);

        return applyConvolution(result, kernel);
    }

    /// <summary>
    /// Creates Gaussian kernel for blurring
    /// </summary>
    /// <param name="size">Kernel size</param>
    /// <param name="sigma">Standard deviation</param>
    /// <returns>Gaussian kernel</returns>
    cv::Mat createGaussianKernel(int size, float sigma) const
    {
        cv::Mat kernel(size, size, CV_32F);
        int center = size / 2;
        float sum = 0.0f;

        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) 
            {
                float dx = x - center;
                float dy = y - center;
                float value = exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel.at<float>(y, x) = value;
                sum += value;
            }
        kernel /= sum;
        return kernel;
    }

    /// <summary>
    /// Applies convolution with image border handling
    /// </summary>
    /// <param name="input">Input image</param>
    /// <param name="kernel">Convolution kernel</param>
    /// <returns>Convolution result</returns>
    cv::Mat applyConvolution(const cv::Mat& input, const cv::Mat& kernel) const
    {
        cv::Mat result = cv::Mat::zeros(input.size(), input.type());
        int kernelRadius = kernel.rows / 2;
        int pixelsProcessed = 0;
        int edgePixels = 0;

        for (int y = 0; y < input.rows; y++)
            for (int x = 0; x < input.cols; x++) 
            {
                float sum = 0.0f, weightSum = 0.0f;
                bool isEdgePixel = false;

                for (int ky = -kernelRadius; ky <= kernelRadius; ky++)
                    for (int kx = -kernelRadius; kx <= kernelRadius; kx++) 
                    {
                        int srcY = y + ky, srcX = x + kx;

                        if (srcY >= 0 && srcY < input.rows && srcX >= 0 && srcX < input.cols) 
                        {
                            float pixel = input.at<float>(srcY, srcX);
                            float weight = kernel.at<float>(ky + kernelRadius, kx + kernelRadius);
                            sum += pixel * weight;
                            weightSum += weight;
                        }
                        else
                            isEdgePixel = true;
                    }

                if (weightSum > 0.0f)
                    result.at<float>(y, x) = sum / weightSum;
                else
                    result.at<float>(y, x) = input.at<float>(y, x);

                pixelsProcessed++;
                if (isEdgePixel)
                    edgePixels++;
            }
        return result;
    }

    /// <summary>
    /// Creates enhanced shadow mask
    /// </summary>
    /// <param name="luminance">Luminance matrix</param>
    /// <returns>Shadow mask</returns>
    cv::Mat createAdvancedShadowMask(const cv::Mat& luminance) const
    {
        float shadowThreshold = 0.4f * tonalWidth;
        cv::Mat smoothMask(luminance.size(), CV_32F);

        for (int y = 0; y < luminance.rows; y++)
            for (int x = 0; x < luminance.cols; x++) {
                float lum = luminance.at<float>(y, x);

                if (lum <= shadowThreshold * 0.6f)
                    smoothMask.at<float>(y, x) = 1.0f;
                else if (lum <= shadowThreshold) 
                {
                    float t = (lum - shadowThreshold * 0.6f) / (shadowThreshold * 0.4f);
                    smoothMask.at<float>(y, x) = 1.0f - t * 0.5f;
                }
                else
                    smoothMask.at<float>(y, x) = 0.0f;
            }

        if (blurRadius > 0.1f) 
        {
            float effectiveRadius = blurRadius * 1.5f;
            smoothMask = applyFastGaussianBlur(smoothMask, effectiveRadius);
        }

        double minVal, maxVal;
        cv::minMaxLoc(smoothMask, &minVal, &maxVal);
        if (maxVal > 0)
            smoothMask /= maxVal;

        return smoothMask;
    }

    /// <summary>
    /// Creates enhanced highlight mask
    /// </summary>
    /// <param name="luminance">Luminance matrix</param>
    /// <returns>Highlight mask</returns>
    cv::Mat createAdvancedHighlightMask(const cv::Mat& luminance) const
    {
        float highlightThreshold = 1.0f - 0.5f * tonalWidth;

        cv::Mat smoothMask(luminance.size(), CV_32F);

        for (int y = 0; y < luminance.rows; y++)
            for (int x = 0; x < luminance.cols; x++) 
            {
                float lum = luminance.at<float>(y, x);

                if (lum >= highlightThreshold)
                    smoothMask.at<float>(y, x) = 1.0f;
                else if (lum >= highlightThreshold * 0.9f) 
                {
                    float t = (lum - highlightThreshold * 0.9f) / (highlightThreshold * 0.1f);
                    smoothMask.at<float>(y, x) = t;
                }
                else
                    smoothMask.at<float>(y, x) = 0.0f;
            }

        if (blurRadius > 0.1f)
            smoothMask = applyFastGaussianBlur(smoothMask, std::min(blurRadius, 20.0f));

        double minVal, maxVal;
        cv::minMaxLoc(smoothMask, &minVal, &maxVal);
        if (maxVal > 0)
            smoothMask /= maxVal;

        return smoothMask;
    }

    /// <summary>
    /// Applies fast Gaussian blur with iterations
    /// </summary>
    /// <param name="input">Input image</param>
    /// <param name="radius">Blur radius</param>
    /// <returns>Blurred image</returns>
    cv::Mat applyFastGaussianBlur(const cv::Mat& input, float radius) const
    {
        if (radius < 1.0f) 
            return input.clone();

        cv::Mat result = input.clone();

        int iterations;
        float iterRadius;

        if (radius <= 8.0f) 
        {
            iterations = 1;
            iterRadius = radius;
        }
        else if (radius <= 20.0f) 
        {
            iterations = 2;
            iterRadius = radius / 2.0f;
        }
        else 
        {
            iterations = 3;
            iterRadius = radius / 3.0f;
        }

        for (int i = 0; i < iterations; i++)
            result = applyGaussianBlur(result, iterRadius);

        return result;
    }
};