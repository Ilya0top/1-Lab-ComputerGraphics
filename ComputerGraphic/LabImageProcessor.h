#pragma once

#include <opencv.hpp>
#include <vector>

/// <summary>
/// Класс для работы с каналами Lab изображений
/// 
/// Предоставляет методы для разделения и объединения каналов
/// Lab изображений. L - яркость, a - зелено-красный, b - сине-желтый.
/// </summary>
class LabImageProcessor 
{
public:

    /// <summary>
    /// Разделяет Lab изображение на отдельные каналы
    /// </summary>
    /// <param name="labImage">Входное Lab изображение (CV_32FC3)</param>
    /// <returns>Вектор из трех матриц: L, a, b каналы</returns>
    /// <exception cref="std::invalid_argument">Если изображение не 3-канальное</exception>
    static std::vector<cv::Mat> splitLab(const cv::Mat& labImage) 
    {
        if (labImage.channels() != 3)
            throw std::invalid_argument("Lab изображение должно иметь 3 канала");

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
    /// Объединяет отдельные каналы в Lab изображение
    /// </summary>
    /// <param name="channels">Вектор из трех матриц: L, a, b каналы</param>
    /// <returns>Объединенное Lab изображение (CV_32FC3)</returns>
    /// <exception cref="std::invalid_argument">Если не 3 канала или разные размеры</exception>
    static cv::Mat mergeLab(const std::vector<cv::Mat>& channels) 
    {
        if (channels.size() != 3)
            throw std::invalid_argument("Для Lab нужно 3 канала");

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