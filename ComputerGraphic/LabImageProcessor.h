#pragma once

#include <opencv.hpp>
#include <vector>

/// <summary>
/// ����� ��� ������ � �������� Lab �����������
/// 
/// ������������� ������ ��� ���������� � ����������� �������
/// Lab �����������. L - �������, a - ������-�������, b - ����-������.
/// </summary>
class LabImageProcessor 
{
public:

    /// <summary>
    /// ��������� Lab ����������� �� ��������� ������
    /// </summary>
    /// <param name="labImage">������� Lab ����������� (CV_32FC3)</param>
    /// <returns>������ �� ���� ������: L, a, b ������</returns>
    /// <exception cref="std::invalid_argument">���� ����������� �� 3-���������</exception>
    static std::vector<cv::Mat> splitLab(const cv::Mat& labImage) 
    {
        if (labImage.channels() != 3)
            throw std::invalid_argument("Lab ����������� ������ ����� 3 ������");

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
    /// ���������� ��������� ������ � Lab �����������
    /// </summary>
    /// <param name="channels">������ �� ���� ������: L, a, b ������</param>
    /// <returns>������������ Lab ����������� (CV_32FC3)</returns>
    /// <exception cref="std::invalid_argument">���� �� 3 ������ ��� ������ �������</exception>
    static cv::Mat mergeLab(const std::vector<cv::Mat>& channels) 
    {
        if (channels.size() != 3)
            throw std::invalid_argument("��� Lab ����� 3 ������");

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