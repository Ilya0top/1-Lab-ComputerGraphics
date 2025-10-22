#pragma once

#include <opencv.hpp>
#include <iostream>
#include <string>
#include "ShadowHighlightsFilter.h"

/// <summary>
/// ����� ��� ������������ ������� Shadow/Highlights
/// </summary>
class TestRunner
{
public:

    /// <summary>
    /// ��������� comprehensive ������������ �������
    /// </summary>
    /// <param name="originalImage">�������� ����������� ��� ������������</param>
    static void runComprehensiveTest(const cv::Mat& originalImage)
    {
        std::cout << "==========================================\n������������ SHADOW/HIGHLIGHTS FILTER\n==========================================\n";

        analyzeImage(originalImage, "");

        // ���� 1: ������ ���������� ����������
        std::cout << "\n--- ���� 1: ������ ��������� ��������� ---\n";

        // ������ 1: ������ ���������� �����
        std::cout << "\n1. ������ ���������� ����� (50%)\n";
        ShadowHighlightsFilter filter1(0.5f, 0.0f);
        cv::Mat result1 = filter1.apply(originalImage);

        // ������ 2: ������ ���������� ������
        std::cout << "\n2. ������ ���������� ������ (40%)\n";
        ShadowHighlightsFilter filter2(0.0f, 0.4f);
        cv::Mat result2 = filter2.apply(originalImage);

        // ������ 3: ��������������� ���������
        std::cout << "\n3. ��������������� ��������� (30% ����, 20% �����)\n";
        ShadowHighlightsFilter filter3(0.3f, 0.2f);
        cv::Mat result3 = filter3.apply(originalImage);

        // ������ 4: ������� ���������
        std::cout << "\n4. ������� ��������� (70% ����, 50% �����)\n";
        ShadowHighlightsFilter filter4(0.7f, 0.5f);
        cv::Mat result4 = filter4.apply(originalImage);

        // ���� 2: ������������ �����������
        std::cout << "\n--- ���� 2: ������������ ����������� ---\n";

        cv::Mat finalDisplay = createComparisonMosaic(originalImage, result1, result2, result3, result4);
        cv::imshow("Shadow/Highlights Filter - ��������� �����������", finalDisplay);

        // ���� 3: ������ ���������� ��������
        std::cout << "\n--- ���� 3: ������ ���������� �������� ---\n";
        analyzePixels(originalImage, result3);

        // ���� 4: ���������� �����������
        std::cout << "\n--- ���� 4: ���������� ����������� ---\n";

        cv::imwrite("Image\\original.jpg", originalImage);
        cv::imwrite("ImageResult\\result_shadows_50.jpg", result1);
        cv::imwrite("ImageResult\\result_highlights_40.jpg", result2);
        cv::imwrite("ImageResult\\result_both_30_20.jpg", result3);
        cv::imwrite("ImageResult\\result_strong_70_50.jpg", result4);
        cv::imwrite("ImageResult\\comparison.jpg", finalDisplay);

        std::cout << "���������� ��������� � �����:\n - Image\\original.jpg\n - ImageResult\\result_shadows_50.jpg\n - ImageResult\\result_highlights_40.jpg\n - ImageResult\\result_both_30_20.jpg\n - ImageResult\\result_strong_70_50.jpg\n - ImageResult\\comparison.jpg\n";

        std::cout << "\n==========================================\n������������ ���������\n������� ����� ������� ��� ������...\n==========================================\n";

        cv::waitKey(0);
    }

    /// <summary>
    /// ��������� ���������������� ���� � ������� �����������
    /// </summary>
    /// <param name="image">�������� �����������</param>
    static void runOptimizedTest(const cv::Mat& image)
    {
        std::cout << "��������� ���� � ������������\n";

        ShadowHighlightsFilter optimalFilter(0.2f, 0.2f, 0.4f, 10.0f);
        cv::Mat optimalResult = optimalFilter.apply(image);

        std::cout << "\n--- ��������� ����������� ---\n";
        analyzePixels(image, optimalResult);

        cv::imwrite("final_optimal_result.jpg", optimalResult);
        std::cout << "\n��������� ��������� �������� � 'final_optimal_result.jpg'\n";

        cv::imshow("��������", image);
        cv::imshow("��������� ��������� (����������� ���������)", optimalResult);

        cv::waitKey(0);
    }

    /// <summary>
    /// ����������� � ������� ���������� �� �����������
    /// </summary>
    /// <param name="image">����������� ��� �������</param>
    static void analyzeImage(const cv::Mat& image, const std::string& imagePath)
    {
        if (!imagePath.empty())
            std::cout << "��������� ����������� �� ����: " << imagePath << std::endl;

        std::cout << "������: " << image.cols << "x" << image.rows << "\n������: " << image.channels() << "\n���: " << image.type() << "\n������ � ������: " << image.total() * image.elemSize() << " ����"  << "\n��� (step): " << image.step << " ���� �� ������"  << "\n\n������� ��������:" << std::endl;
        
        for (int y = 0; y < std::min(3, image.rows); y++)
            for (int x = 0; x < std::min(3, image.cols); x++)
                if (image.channels() == 3) 
                {
                    cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                    std::cout << "�������(" << x << "," << y << "): " << "B=" << (int)pixel[0] << ", G=" << (int)pixel[1] << ", R=" << (int)pixel[2] << std::endl;
                }
    }

private:
    /// <summary>
    /// ����������� ���������� ������� �����������
    /// </summary>
    static void analyzePixels(const cv::Mat& originalImage, const cv::Mat& resultImage)
    {
        std::vector<cv::Point> testPoints = {
            cv::Point(100, 100),
            cv::Point(50, 200),  
            cv::Point(400, 250)
        };

        for (const cv::Point& pt : testPoints)
            if (pt.x < originalImage.cols && pt.y < originalImage.rows) 
            {
                cv::Vec3b origPixel = originalImage.at<cv::Vec3b>(pt.y, pt.x);
                cv::Vec3b resultPixel = resultImage.at<cv::Vec3b>(pt.y, pt.x);

                std::cout << "������� (" << pt.x << ", " << pt.y << "):" << "\n��������:  B=" << (int)origPixel[0] << ", G=" << (int)origPixel[1] << ", R=" << (int)origPixel[2] << "\n���������: B=" << (int)resultPixel[0] << ", G=" << (int)resultPixel[1] << ", R=" << (int)resultPixel[2] << std::endl;

                float origBrightness = calculateBrightness(origPixel);
                float resultBrightness = calculateBrightness(resultPixel);
                std::cout << "  �������: " << origBrightness << " -> " << resultBrightness << " (���������: " << (resultBrightness - origBrightness) << ")" << std::endl;
            }
    }

    /// <summary>
    /// ������� ������� ��� ����������� ���������
    /// </summary>
    static cv::Mat createComparisonMosaic(const cv::Mat& original, const cv::Mat& result1, const cv::Mat& result2, const cv::Mat& result3, const cv::Mat& result4)
    {
        int displayWidth = 600;
        int displayHeight = 400;

        cv::Mat displayOriginal, displayResult1, displayResult2, displayResult3, displayResult4;
        cv::resize(original, displayOriginal, cv::Size(displayWidth, displayHeight));
        cv::resize(result1, displayResult1, cv::Size(displayWidth, displayHeight));
        cv::resize(result2, displayResult2, cv::Size(displayWidth, displayHeight));
        cv::resize(result3, displayResult3, cv::Size(displayWidth, displayHeight));
        cv::resize(result4, displayResult4, cv::Size(displayWidth, displayHeight));

        cv::putText(displayOriginal, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(displayResult1, "Shadows 50%", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(displayResult2, "Highlights 40%", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(displayResult3, "Both 30%/20%", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(displayResult4, "Both 70%/50%", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        cv::Mat topRow, bottomRow, finalDisplay;
        cv::hconcat(displayOriginal, displayResult1, topRow);
        cv::hconcat(displayResult2, displayResult3, bottomRow);
        cv::vconcat(topRow, bottomRow, finalDisplay);

        return finalDisplay;
    }

    /// <summary>
    /// ��������� ������� �������
    /// </summary>
    static float calculateBrightness(const cv::Vec3b& pixel)
    {
        return 0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0];
    }
};