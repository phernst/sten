#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace sten
{

namespace indexing
{

constexpr static auto None = std::nullopt;

struct Slice {
    using Member = std::optional<std::int64_t>;
    Member start{None};
    Member stop{None};
    Member step{None};

    [[nodiscard]] constexpr Slice apply(const Slice& other) const
    {
        constexpr auto toStepInt = [](const Member& member) {
            return member.value_or(1);
        };
        constexpr auto toStartInt = [](const Member& member) {
            return member.value_or(0);
        };
        constexpr auto toStopInt = [](const Member& member) {
            return member.value_or(0);
        };

        const auto newStep = (!step && !other.step)
                                 ? None
                                 : Member{toStepInt(step) * toStepInt(other.step)};
        const auto newStart = (!start && !other.start)
                                  ? None
                                  : Member{toStartInt(start) + toStepInt(step) * toStartInt(other.start)};
        const auto newStop = other.stop.has_value()
                                 ? toStartInt(start) + toStepInt(step) * toStopInt(other.stop)
                                 : stop;
        return {newStart, newStop, newStep};
    }
};

} // namespace indexing

namespace detail
{

struct TensorData {
    std::vector<float> buffer;
    std::vector<std::int64_t> dimensions;
};

} // namespace detail

class Tensor;
Tensor arange(int end);

class Tensor
{
public:
    [[nodiscard]] Tensor index(const std::vector<indexing::Slice>& indices) const
    {
        // create the new dimensions
        std::vector<std::int64_t> newDimensions(m_dimensions.size());
        std::transform(
            cbegin(m_dimensions), cend(m_dimensions), cbegin(indices),
            begin(newDimensions),
            [](const auto& oldDim, const auto& dimIndices) {
                const auto stop = dimIndices.stop.value_or(oldDim);
                const auto start = dimIndices.start.value_or(0);
                const auto step = dimIndices.step.value_or(1);
                return (stop - start + step - 1) / step;
            });

        // find the new start position to be able to determine the new buffer offset
        std::vector<std::int64_t> startPos(m_dimensions.size());
        std::transform(cbegin(indices), cend(indices), begin(startPos), [](const auto& dimIndices) {
            return dimIndices.start.value_or(0);
        });

        // now determine the new buffer offset
        const auto newOffset = std::transform_reduce(
                                   cbegin(startPos), cend(startPos), cbegin(m_strides), 0) +
                               m_offset;

        // finally, determine the new strides
        std::vector<std::ptrdiff_t> newStrides(m_dimensions.size());
        std::transform(cbegin(m_strides), cend(m_strides), cbegin(indices), begin(newStrides),
                       [](const auto& dimStride, const auto& dimIndices) {
                           return dimStride * dimIndices.step.value_or(1);
                       });

        return {m_data, newOffset, newDimensions, newStrides};
    }

    void print() const
    {
        for (int i = 0; i < m_dimensions[0]; i++) {
            std::cout << m_data->buffer[m_offset + i * m_strides[0]] << ", ";
        }
        std::cout << '\n';
    }

private:
    Tensor(std::shared_ptr<detail::TensorData> data, std::ptrdiff_t offset, std::vector<std::int64_t> dimensions, std::vector<std::ptrdiff_t> strides)
    : m_data{data}
    , m_offset{offset}
    , m_dimensions{std::move(dimensions)}
    , m_strides{std::move(strides)}
    {
    }
    std::shared_ptr<detail::TensorData> m_data;
    std::ptrdiff_t m_offset;
    std::vector<std::int64_t> m_dimensions;
    std::vector<std::ptrdiff_t> m_strides;

    friend Tensor arange(int);
};

[[nodiscard]] inline Tensor arange(int end)
{
    auto data = std::make_shared<detail::TensorData>(
        std::vector<float>(end), std::vector<std::int64_t>{static_cast<std::int64_t>(end)});

    std::iota(data->buffer.begin(), data->buffer.end(), 0.f);
    return {data, 0, data->dimensions, {1}};
}

} // namespace sten