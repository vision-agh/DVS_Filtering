//
// Created by k1234 on 19/06/2023.
//

#ifndef NOISEGENERATOR_NOISE_GENERATOR_ALGORITHM_H
#define NOISEGENERATOR_NOISE_GENERATOR_ALGORITHM_H

#include <cmath>
#include <vector>
#include <random>
#include <ostream>

#include "event2d.h"
// #include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"

typedef std::double_t ComputationType;

namespace Metavision {

    class RandomGenerator {
    public:
        RandomGenerator(std::uint16_t width, std::uint16_t height) {
            uniformGeneratorWidth = std::uniform_int_distribution<std::uint32_t>(0, width - 1);
            uniformGeneratorHeight = std::uniform_int_distribution<std::uint32_t>(0, height - 1);
        }

        std::uint16_t generateWidth() {
            return (std::uint16_t) uniformGeneratorWidth(gen);
        }

        std::uint16_t generateHeight() {
            return (std::uint16_t) uniformGeneratorHeight(gen);
        }

        ComputationType generateGauss() {
            return gaussGenerator(gen);
        }

        ComputationType generateUniform() {
            return uniformGenerator(gen);
        }

    private:
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<std::uint32_t> uniformGeneratorWidth;
        std::uniform_int_distribution<std::uint32_t> uniformGeneratorHeight;
        std::normal_distribution<ComputationType> gaussGenerator{0, 1};
        std::uniform_real_distribution<ComputationType> uniformGenerator{0, 1};
    };

    /// @brief Class that performs the noise generation algorithm
    class NoiseGeneratorAlgorithm {
    public:

        /// @brief Creates a NoiseGeneratorAlgorithm class
        inline explicit NoiseGeneratorAlgorithm(std::uint32_t width = 1280, std::uint32_t height = 720,
                                                ComputationType shotNoiseRateHz = 0,
                                                ComputationType poissonDivider = 30,
                                                uint32_t timestampResolutionUs = 1)
                : width_(width), height_(height), shotNoiseRateHz_(shotNoiseRateHz),
                  poissonDivider_(poissonDivider), timestampResolutionUs_(timestampResolutionUs),
                  noiseRateArray_(std::vector<ComputationType>(width * height)),
                  noiseRateIntervals_(std::vector<ComputationType>(width * height)),
                  randomGenerator_(RandomGenerator(width, height)) {
            calculateProbThr();
        };

        /// @brief Default destructor
        inline ~NoiseGeneratorAlgorithm() = default;

        /// @brief Applies the noise generator to the given input buffer storing the result in the output buffer
        /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
        /// or equivalent
        /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
        /// or equivalent
        /// @param it_begin Iterator to first input event
        /// @param it_end Iterator to the past-the-end event
        /// @param inserter Output iterator or back inserter
        /// @return Iterator pointing to the past-the-end event added in the output
        template<class InputIt, class OutputIt>
        inline OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

    private:
        /// @brief Parameters of the noise generator algorithm
        std::uint16_t width_;
        std::uint16_t height_;
        ComputationType shotNoiseRateHz_;
        ComputationType poissonDtUs_;
        ComputationType shotOffThresholdProb_;
        ComputationType shotOnThresholdProb_;
        ComputationType poissonDivider_;

        /// @brief Current state of the noise generator algorithm
        std::uint32_t lastTimestampPreviousPacket_ = 0;
        std::vector<ComputationType> noiseRateArray_;
        std::vector<ComputationType> noiseRateIntervals_;
        RandomGenerator randomGenerator_;
        uint32_t timestampResolutionUs_;

        /// @brief Calculate the probabilities of noise generation
        void calculateProbThr();

        /// @brief Add the noise events to the output stream
        template<class InputIt, class OutputIt>
        void addNoise(InputIt it_begin, InputIt it_end, OutputIt inserter);

        /// @brief Generate noise between two timestamps
        template<class OutputIt>
        void insertNoiseEvents(std::uint32_t previousTimestamp, std::uint32_t currentTimestamp, OutputIt inserter);

        /// @brief Generate noise for the particular timestamp
        template<class OutputIt>
        void sampleNoiseEvent(std::uint32_t timestamp, OutputIt inserter);

        /// @brief Generate shot event of the particular timestamp and polarity
        template<class OutputIt>
        void injectShotNoiseEvent(std::uint32_t timestamp, std::uint8_t polarity, OutputIt inserter);
    };
}

template<class InputIt, class OutputIt>
inline OutputIt
Metavision::NoiseGeneratorAlgorithm::process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {

    if (it_begin == it_end) {
        return inserter;
    }

    Event2d firstEvent = *it_begin;

    addNoise(it_begin, it_end, inserter);
    lastTimestampPreviousPacket_ = (it_end - 1)->t;
    return inserter;
}

void Metavision::NoiseGeneratorAlgorithm::calculateProbThr() {
    std::uint32_t npix = width_ * height_;
    ComputationType rate = 1.0f / (shotNoiseRateHz_) / (ComputationType) npix;
    poissonDtUs_ = 1e6f * rate / poissonDivider_;
    ComputationType minPoissonDtUs = timestampResolutionUs_ / 50.0;
    ComputationType maxPoissonDtUs = timestampResolutionUs_ / 2.0;
    if (poissonDtUs_ < minPoissonDtUs)
        poissonDtUs_ = minPoissonDtUs;
    else if (poissonDtUs_ > maxPoissonDtUs)
        poissonDtUs_ = maxPoissonDtUs;
    else {
        poissonDtUs_ = timestampResolutionUs_ / std::round(timestampResolutionUs_ / poissonDtUs_);
    }
    shotOffThresholdProb_ = poissonDtUs_ * 0.5e-6f * (ComputationType) npix * shotNoiseRateHz_;
    shotOnThresholdProb_ = 1.0f - shotOffThresholdProb_;
}

template<class InputIt, class OutputIt>
void Metavision::NoiseGeneratorAlgorithm::addNoise(InputIt it_begin, InputIt it_end, OutputIt inserter) {
    if (shotNoiseRateHz_ == 0) {
        std::copy(it_begin, it_end, inserter);
        return;
    }

    std::uint32_t firstTs = it_begin->t;
    if (lastTimestampPreviousPacket_ > 0) {
        std::uint32_t lastPacketTs = lastTimestampPreviousPacket_;
        insertNoiseEvents(lastPacketTs, firstTs, inserter);
    }

    std::uint32_t previousTimestamp = 0;
    std::uint32_t lastTimestamp = it_begin->t;
    for (auto it = it_begin; it != it_end; it++) {
        if (it->t == firstTs) {
            inserter = *it;
            continue;
        }
        previousTimestamp = lastTimestamp;
        lastTimestamp = it->t;
        insertNoiseEvents(previousTimestamp, lastTimestamp, inserter);
        inserter = *it;
    }
}

template<class OutputIt>
void Metavision::NoiseGeneratorAlgorithm::insertNoiseEvents(std::uint32_t previousTimestamp,
                                                            std::uint32_t currentTimestamp,
                                                            OutputIt inserter) {
    auto iterations = (std::uint32_t) std::round((currentTimestamp - previousTimestamp) / poissonDtUs_);
    auto ts = (std::double_t) previousTimestamp;
    for (std::uint32_t idx = 0; idx < iterations; ++idx) {
        std::uint32_t TsResolution = (std::uint32_t) ts - (std::uint32_t) ts % timestampResolutionUs_;
        sampleNoiseEvent((std::uint32_t) TsResolution, inserter);
        ts += poissonDtUs_;
    }
}

template<class OutputIt>
void Metavision::NoiseGeneratorAlgorithm::sampleNoiseEvent(std::uint32_t timestamp, OutputIt inserter) {
    ComputationType randomVar = randomGenerator_.generateUniform();
    if (randomVar < shotOffThresholdProb_) {
        injectShotNoiseEvent(timestamp, 2, inserter);
    } else if (randomVar > shotOnThresholdProb_) {
        injectShotNoiseEvent(timestamp, 3, inserter);
    }
}

template<class OutputIt>
void
Metavision::NoiseGeneratorAlgorithm::injectShotNoiseEvent(std::uint32_t timestamp, std::uint8_t polarity, OutputIt inserter) {
    Event2d shotEvent(0, 0, polarity, timestamp);
    shotEvent.x = randomGenerator_.generateWidth();
    shotEvent.y = randomGenerator_.generateHeight();
    inserter = shotEvent;
}

#endif //NOISEGENERATOR_NOISE_GENERATOR_ALGORITHM_H