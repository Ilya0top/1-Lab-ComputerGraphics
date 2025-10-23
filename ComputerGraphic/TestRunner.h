#pragma once

#include <opencv.hpp>
#include <iostream>
#include <string>
#include "ShadowHighlightsFilter.h"

/// <summary>
/// Class for testing Shadow/Highlights filter
/// </summary>
class TestRunner
{
public:

    /// <summary>
    /// Runs comprehensive testing of the filter
    /// </summary>
    /// <param name="originalImage">Original image for testing</param>
    static void runComprehensiveTest(const cv::Mat& originalImage)
    {
        std::cout << "==========================================\nSHADOW/HIGHLIGHTS FILTER TESTING\n==========================================\n";
        
        analyzeImage(originalImage, "");

        // Test 1: Different parameter combinations
        std::cout << "\n--- TEST 1: Different correction parameters ---\n";

        // Case 1: Only shadow lightening
        std::cout << "\n1. Shadow lightening only (50%)\n";
        ShadowHighlightsFilter filter1(0.5f, 0.0f);
        cv::Mat result1 = filter1.apply(originalImage);

        // Case 2: Only highlight darkening
        std::cout << "\n2. Highlight darkening only (40%)\n";
        ShadowHighlightsFilter filter2(0.0f, 0.4f);
        cv::Mat result2 = filter2.apply(originalImage);

        // Case 3: Combined correction
        std::cout << "\n3. Combined correction (30% shadows, 20% highlights)\n";
        ShadowHighlightsFilter filter3(0.3f, 0.2f);
        cv::Mat result3 = filter3.apply(originalImage);

        // Case 4: Strong correction
        std::cout << "\n4. Strong correction (70% shadows, 50% highlights)\n";
        ShadowHighlightsFilter filter4(0.7f, 0.5f);
        cv::Mat result4 = filter4.apply(originalImage);

        // Test 2: Results visualization
        std::cout << "\n--- TEST 2: Results visualization ---\n";

        cv::Mat finalDisplay = createComparisonMosaic(originalImage, result1, result2, result3, result4);
        cv::imshow("Shadow/Highlights Filter - Results Comparison", finalDisplay);

        // Test 3: Specific areas analysis
        std::cout << "\n--- TEST 3: Specific pixels analysis ---\n";
        analyzePixels(originalImage, result3);

        // Test 4: Results saving
        std::cout << "\n--- TEST 4: Results saving ---\n";

        cv::imwrite("Image\\original.jpg", originalImage);
        cv::imwrite("ImageResult\\result_shadows_50.jpg", result1);
        cv::imwrite("ImageResult\\result_highlights_40.jpg", result2);
        cv::imwrite("ImageResult\\result_both_30_20.jpg", result3);
        cv::imwrite("ImageResult\\result_strong_70_50.jpg", result4);
        cv::imwrite("ImageResult\\comparison.jpg", finalDisplay);

        std::cout << "Results saved to files:\n - Image\\original.jpg\n - ImageResult\\result_shadows_50.jpg\n - ImageResult\\result_highlights_40.jpg\n - ImageResult\\result_both_30_20.jpg\n - ImageResult\\result_strong_70_50.jpg\n - ImageResult\\comparison.jpg\n";

        std::cout << "\n==========================================\nTESTING COMPLETED\nPress any key to exit...\n==========================================\n";
        
        cv::waitKey(0);
    }

    /// <summary>
    /// Runs optimized test with best parameters
    /// </summary>
    /// <param name="image">Original image</param
    static void runOptimizedTest(const cv::Mat& image)
    {
        std::cout << "FINAL TEST WITH OPTIMIZATION\n";

        ShadowHighlightsFilter optimalFilter(0.2f, 0.2f, 0.4f, 10.0f);
        cv::Mat optimalResult = optimalFilter.apply(image);

        std::cout << "\n--- RESULTS COMPARISON ---\n";
        analyzePixels(image, optimalResult);

        cv::imwrite("final_optimal_result.jpg", optimalResult);
        std::cout << "\nFinal result saved to 'final_optimal_result.jpg'\n";

        cv::imshow("Original", image);
        cv::imshow("Final result (optimal settings)", optimalResult);

        cv::waitKey(0);
    }

    /// <summary>
    /// Analyzes and displays image information
    /// </summary>
    /// <param name="image">Image to analyze</param>
    /// <param name="imagePath">Image file path</param>
    static void analyzeImage(const cv::Mat& image, const std::string& imagePath)
    {
        if (!imagePath.empty())
            std::cout << "Image loaded from path: " << imagePath << std::endl;

        std::cout << "Size: " << image.cols << "x" << image.rows << "\nChannels: " << image.channels() << "\nType: " << image.type() << "\nSize in bytes: " << image.total() * image.elemSize() << " bytes" << "\nStep: " << image.step << " bytes per row" << "\n\nSample pixels:" << std::endl;

        for (int y = 0; y < std::min(3, image.rows); y++)
            for (int x = 0; x < std::min(3, image.cols); x++)
                if (image.channels() == 3) 
                {
                    cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                    std::cout << "Pixel(" << x << "," << y << "): " << "B=" << (int)pixel[0] << ", G=" << (int)pixel[1] << ", R=" << (int)pixel[2] << std::endl;
                }
    }

private:
    /// <summary>
    /// Analyzes specific image pixels
    /// </summary>
    /// <param name="originalImage">Original image</param>
    /// <param name="resultImage">Processed image</param>
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

                std::cout << "Pixel (" << pt.x << ", " << pt.y << "):" << "\nOriginal:  B=" << (int)origPixel[0] << ", G=" << (int)origPixel[1] << ", R=" << (int)origPixel[2] << "\nResult:    B=" << (int)resultPixel[0] << ", G=" << (int)resultPixel[1] << ", R=" << (int)resultPixel[2] << std::endl;
                float origBrightness = calculateBrightness(origPixel);
                float resultBrightness = calculateBrightness(resultPixel);
                std::cout << "  Brightness: " << origBrightness << " -> " << resultBrightness << " (change: " << (resultBrightness - origBrightness) << ")" << std::endl;
            }
    }

    /// <summary>
    /// Creates mosaic for visual comparison
    /// </summary>
    /// <param name="original">Original image</param>
    /// <param name="result1">First result image</param>
    /// <param name="result2">Second result image</param>
    /// <param name="result3">Third result image</param>
    /// <param name="result4">Fourth result image</param>
    /// <returns>Comparison mosaic image</returns
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
    /// Calculates pixel brightness
    /// </summary>
    /// <param name="pixel">Pixel value</param>
    /// <returns>Brightness value</returns>
    static float calculateBrightness(const cv::Vec3b& pixel)
    {
        return 0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0];
    }
};