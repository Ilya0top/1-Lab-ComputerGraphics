#pragma once

#include <opencv.hpp>
#include <vector>

/// <summary>
/// The class for working with Lab Image channels
///
/// Provides methods for separating and combining Lab image channels. 
/// L - brightness, a - green-red, b - blue-yellow.
/// </summary>
class LabImageProcessor 
{
public:

    /// <summary>
    /// Splits the Lab image into separate channels    
    /// </summary>
    /// <param name="labImage">Input Lab Image (CV_32FC3)</param>
    /// <returns>A vector of three matrices: L, a, b channels</returns>
    /// <exception cref="std::invalid_argument">If the image is not 3-channel</exception>
    static std::vector<cv::Mat> splitLab(const cv::Mat& labImage) 
    {
        if (labImage.channels() != 3)
            throw std::invalid_argument("Lab image must have 3 channels");

        std::vector<cv::Mat> channels(3);
        channels[0] = cv::Mat(labImage.size(), CV_32F);
        channels[1] = cv::Mat(labImage.size(), CV_32F);
        channels[2] = cv::Mat(labImage.size(), CV_32F); 

        for (int y = 0; y < labImage.rows; y++)
            for (int x = 0; x < labImage.cols; x++) 
            {
                cv::Vec3f labPixel = labImage.at<cv::Vec3f>(y, x);

                channels[0].at<float>(y, x) = labPixel[0];
                channels[1].at<float>(y, x) = labPixel[1];
                channels[2].at<float>(y, x) = labPixel[2];
            }

        return channels;
    }

    /// <summary>
    /// Combines individual channels into a Lab image
    /// </summary>
    /// <param name="channels">A vector of three matrices: L, a, b channels</param>
    /// <returns>Combined Lab Image (CV_32FC3)</returns>
    /// <exception cref="std::invalid_argument">If not 3 channels or different sizes</exception>
    static cv::Mat mergeLab(const std::vector<cv::Mat>& channels) 
    {
        if (channels.size() != 3)
            throw std::invalid_argument("Lab needs 3 channels");

        cv::Mat labImage(channels[0].size(), CV_32FC3);

        for (int y = 0; y < labImage.rows; y++)
            for (int x = 0; x < labImage.cols; x++) {
                cv::Vec3f labPixel;
                labPixel[0] = channels[0].at<float>(y, x);
                labPixel[1] = channels[1].at<float>(y, x); 
                labPixel[2] = channels[2].at<float>(y, x);

                labImage.at<cv::Vec3f>(y, x) = labPixel;
            }

        return labImage;
    }
};