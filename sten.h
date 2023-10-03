#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <variant>
#include <vector>

namespace sten
{

namespace indexing
{

namespace detail
{
struct NoneType {
};
} // namespace detail

constexpr static detail::NoneType None{};

struct Slice {
    using Member = std::variant<detail::NoneType, std::int64_t>;
    Member start{None};
    Member stop{None};
    Member step{None};

    [[nodiscard]] constexpr Slice apply(const Slice& other) const
    {
        constexpr auto isNone = [](const Member& member) {
            return std::holds_alternative<detail::NoneType>(member);
        };

        constexpr auto toStepInt = [isNone](const Member& member) {
            return isNone(member) ? 1 : std::get<std::int64_t>(member);
        };
        constexpr auto toStartInt = [isNone](const Member& member) {
            return isNone(member) ? 0 : std::get<std::int64_t>(member);
        };
        constexpr auto toStopInt = [isNone](const Member& member) {
            return isNone(member) ? 0 : std::get<std::int64_t>(member);
        };

        const auto newStep = isNone(step) && isNone(other.step)
                                 ? Member{None} // both None -> None
                                 : Member{toStepInt(step) * toStepInt(other.step)};
        const auto newStart = isNone(start) && isNone(other.start)
                                  ? Member{None}
                                  : Member{toStartInt(start) + toStepInt(step) * toStartInt(other.start)};
        const auto newStop = !isNone(other.stop)
                                 ? Member{toStartInt(start) + toStepInt(step) * toStopInt(other.stop)}
                                 : Member{stop};
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
                const auto stop = !std::holds_alternative<std::int64_t>(dimIndices.stop)
                                      ? oldDim
                                      : std::get<std::int64_t>(dimIndices.stop);
                const auto start = !std::holds_alternative<std::int64_t>(dimIndices.start)
                                       ? 0
                                       : std::get<std::int64_t>(dimIndices.start);
                const auto step = !std::holds_alternative<std::int64_t>(dimIndices.step)
                                      ? 1
                                      : std::get<std::int64_t>(dimIndices.step);
                return (stop - start + step - 1) / step;
            });

        // find the new start position to be able to determine the new buffer offset
        std::vector<std::int64_t> startPos(m_dimensions.size());
        std::transform(cbegin(indices), cend(indices), begin(startPos), [](const auto& dimIndices) {
            return !std::holds_alternative<std::int64_t>(dimIndices.start)
                       ? 0
                       : std::get<std::int64_t>(dimIndices.start);
        });

        // now determine the new buffer offset
        const auto newOffset = std::transform_reduce(
                                   cbegin(startPos), cend(startPos), cbegin(m_strides), 0) +
                               m_offset;

        // finally, determine the new strides
        std::vector<std::ptrdiff_t> newStrides(m_dimensions.size());
        std::transform(cbegin(m_strides), cend(m_strides), cbegin(indices), begin(newStrides),
                       [](const auto& dimStride, const auto& dimIndices) {
                           return dimStride * (!std::holds_alternative<std::int64_t>(dimIndices.step)
                                                   ? 1
                                                   : std::get<std::int64_t>(dimIndices.step));
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

[[nodiscard]] Tensor arange(int end)
{
    auto data = std::make_shared<detail::TensorData>(
        std::vector<float>(end), std::vector<std::int64_t>{static_cast<std::int64_t>(end)});

    std::iota(data->buffer.begin(), data->buffer.end(), 0.f);
    return {data, 0, data->dimensions, {1}};
}

} // namespace sten