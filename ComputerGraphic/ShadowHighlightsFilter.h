#pragma once

#include <opencv.hpp>
#include <vector>
#include "ColorConverter.h"
#include "LabImageProcessor.h"
#include <iostream>

/// <summary>
/// ������ ��� ��������� ����� � ������ �����������.
/// 
/// ��������� ���������������� �������� ����������� Adobe Photoshop
/// � ���������� ���������� ����� � ������, �������� ����������
/// � ����������� ��������� �������.
/// </summary>
class ShadowHighlightsFilter
{
	/// <summary>���� ���������� ����� (0.0 - 1.0)</summary>
	float shadowAmount;

    /// <summary>���� ���������� ������ (0.0 - 1.0)</summary>
	float highlightAmount;

    /// <summary>��������� ������ (0.0 - 1.0)</summary>
    float tonalWidth;

    /// <summary>������ �������� ����� (0.0 - 50.0)</summary>
    float blurRadius;

public:

    /// <summary>
    /// ������� ������ � ���������� �����������
    /// </summary>
    /// <param name="shadows">���� ���������� ����� (0.0 - 1.0)</param>
    /// <param name="highlights">���� ���������� ������ (0.0 - 1.0)</param>
    /// <param name="width">��������� ������ (0.0 - 1.0)</param>
    /// <param name="radius">������ �������� ����� (0.0 - 50.0)</param>
    ShadowHighlightsFilter(float shadows = 0.3f, float highlights = 0.3f, float width = 0.5f, float radius = 15.0f)
        :   shadowAmount(std::max(0.0f, std::min(1.0f, shadows))), 
            highlightAmount(std::max(0.0f, std::min(1.0f, highlights))), 
            tonalWidth(std::max(0.0f, std::min(1.0f, width))), 
            blurRadius(std::max(0.0f, std::min(50.0f, radius)))
    {
    }

    /// <summary>
    /// ��������� ������ � �����������
    /// </summary>
    /// <param name="inputImage">������� BGR �����������</param>
    /// <returns>������������ �����������</returns>
    /// <exception cref="std::invalid_argument">���� ������� ����������� ������</exception>
    cv::Mat apply(const cv::Mat& inputImage) 
    {
        if (inputImage.empty())
            throw std::invalid_argument("������� ����������� �����");

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
    /// ������������� ���� ���������� �����
    /// </summary>
    /// <param name="amount">�������� �� 0.0 �� 1.0</param>
    void setShadowAmount(float amount) 
    { 
        shadowAmount = std::max(0.0f, std::min(1.0f, amount)); 
    }

    /// <summary>
    /// ������������� ���� ���������� ������
    /// </summary>
    /// <param name="amount">�������� �� 0.0 �� 1.0</param>
    void setHighlightAmount(float amount) 
    { 
        highlightAmount = std::max(0.0f, std::min(1.0f, amount)); 
    }

    /// <summary>
    /// ������������� ��������� ������
    /// </summary>
    /// <param name="width">�������� �� 0.0 �� 1.0</param>
    void setTonalWidth(float width) 
    { 
        tonalWidth = std::max(0.0f, std::min(1.0f, width)); 
    }

    /// <summary>
    /// ������������� ������ �������� �����
    /// </summary>
    /// <param name="radius">�������� �� 0.0 �� 50.0</param>
    void setBlurRadius(float radius) 
    { 
        blurRadius = std::max(0.0f, std::min(50.0f, radius)); 
    }

    /// <summary>
    /// ������� ������� ��������� �������
    /// </summary
    void printCurrentSettings() const 
    {
        std::cout << "������� ��������� Shadow/Highlights:\nShadow Amount: " << shadowAmount * 100 << "%\nHighlight Amount: " << highlightAmount * 100 << "%\nTonal Width: " << tonalWidth << "\nBlur Radius: " << blurRadius << " px";
    }
private:

	/// <summary>
	/// ����������� �������� ������� � ��������� 0-1
	/// </summary>
	/// <param name="luminance">������� ������� Lab ������</param>
	/// <returns>��������������� ������� �������</returns>
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
    /// ������������� �������� ������� � �������� 0-255
    /// </summary>
    /// <param name="normalizedLuminance">��������������� ������� �������</param>
    /// <returns>����������������� ������� �������</returns>
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
    /// ��������� ����������� ��������� ������� � �������������� �����
    /// </summary>
    /// <param name="luminance">������� �������</param>
    /// <param name="shadowMask">����� �����</param>
    /// <param name="highlightMask">����� ������</param>
    /// <returns>����������������� ������� �������</returns>
    cv::Mat applyAdvancedCorrection(const cv::Mat& luminance, const cv::Mat& shadowMask, const cv::Mat& highlightMask) const
    {
        cv::Mat result = luminance.clone();

        for (int y = 0; y < result.rows; y++)
            for (int x = 0; x < result.cols; x++)
            {
                float lum = luminance.at<float>(y, x), shadow = shadowMask.at<float>(y, x), highlight = highlightMask.at<float>(y, x);

                float shadowCorrection = shadowAmount * shadow * (1.0f - lum) * 0.7f, highlightCorrection = highlightAmount * highlight * lum * 0.7f;

                float corrected = lum + shadowCorrection - highlightCorrection;

                float minVal = lum * 0.3f, maxVal = 1.0f - (1.0f - lum) * 0.3f;

                result.at<float>(y, x) = std::max(minVal, std::min(maxVal, corrected));
            }

        return result;
    }

    /// <summary>
    /// ������������ ������� � BGR �������� ������������
    /// </summary>
    /// <param name="correctedLuminance">����������������� �������</param>
    /// <param name="labChannels">������ Lab �����������</param>
    /// <returns>BGR �����������</returns>
	cv::Mat convertBackToBGR(const cv::Mat& correctedLuminance, std::vector<cv::Mat>& labChannels) 
	{
		cv::Mat denormalizedLuminance = denormalizeLuminance(correctedLuminance);
		labChannels[0] = denormalizedLuminance;
		cv::Mat resultLab = LabImageProcessor::mergeLab(labChannels);
		return ColorConverter::Lab2BGR(resultLab);
	}

    /// <summary>
    /// ��������� �������� �������� � �����������
    /// </summary>
    /// <param name="input">������� �����������</param>
    /// <param name="radius">������ ��������</param>
    /// <returns>�������� �����������</returns>
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
    /// ������� �������� ���� ��� ��������
    /// </summary>
    /// <param name="size">������ ����</param>
    /// <param name="sigma">����������� ����������</param>
    /// <returns>�������� ����</returns>
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
    /// ��������� ������� � ���������� ������ �����������
    /// </summary>
    /// <param name="input">������� �����������</param>
    /// <param name="kernel">���� �������</param>
    /// <returns>��������� �������</returns>
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
    /// ������� ���������� ����� ��� �����
    /// </summary>
    /// <param name="luminance">������� �������</param>
    /// <returns>����� �����</returns>
    cv::Mat createAdvancedShadowMask(const cv::Mat& luminance) const
    {
        float shadowThreshold = 0.5f * tonalWidth;
        cv::Mat smoothMask(luminance.size(), CV_32F);

        for (int y = 0; y < luminance.rows; y++)
            for (int x = 0; x < luminance.cols; x++) {
                float lum = luminance.at<float>(y, x);

                if (lum <= shadowThreshold * 0.3f)
                    smoothMask.at<float>(y, x) = 1.0f;
                else if (lum <= shadowThreshold) 
                {
                    float t = (lum - shadowThreshold * 0.3f) / (shadowThreshold * 0.7f);
                    smoothMask.at<float>(y, x) = 1.0f - t * t;
                }
                else
                    smoothMask.at<float>(y, x) = 0.0f;
            }

        if (blurRadius > 0.1f) 
        {
            float effectiveRadius = std::min(blurRadius, 20.0f);
            smoothMask = applyFastGaussianBlur(smoothMask, effectiveRadius);
        }

        double minVal, maxVal;
        cv::minMaxLoc(smoothMask, &minVal, &maxVal);
        if (maxVal > 0)
            smoothMask /= maxVal;

        return smoothMask;
    }

    /// <summary>
    /// ������� ���������� ����� ��� ������
    /// </summary>
    /// <param name="luminance">������� �������</param>
    /// <returns>����� ������</returns>
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
    /// ��������� ������� �������� �������� � ����������
    /// </summary>
    /// <param name="input">������� �����������</param>
    /// <param name="radius">������ ��������</param>
    /// <returns>�������� �����������</returns>
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