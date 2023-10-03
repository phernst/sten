#pragma once

#include <cstdint>
#include <memory>
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
    [[nodiscard]] Tensor index(const std::vector<indexing::Slice>& indices) const;

    void print() const;

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

} // namespace sten