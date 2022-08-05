module Utils
export one_hot, read_image, MetricAccumulator, EpochEnd

using FileIO, Images
using Statistics: mean

struct EpochEnd end

mutable struct MetricAccumulator
    acc::Vector{Float64}
    metric::Function
    display::Function
end

MetricAccumulator(metric::Function, display::Function) = MetricAccumulator(Vector{Float64}(), metric, display)

function (a::MetricAccumulator)(prediction, ground_truth)
    a.acc = vcat(a.acc, a.metric(prediction, ground_truth))
    return a.acc |> mean |> a.display
end

function (a::MetricAccumulator)(::EpochEnd)
    a.acc = Vector{Float64}()
end

struct MetricAggregator
    accumulators::Vector{MetricAccumulator}
end

function (a::MetricAggregator)(prediction, ground_truth) 
    lines = [accumulator(prediction, ground_truth) for accumulator in a.accumulators]
    return join(lines, ", ")
end

function (a::MetricAggregator)(::EpochEnd) 
    for accumulator in a.accumulators
        accumulator(EpochEnd())
    end
end

function one_hot(class, classes)
    v = zeros(Float32, length(classes))
    for (i, c) in enumerate(classes)
        if c == class
            v[i] = 1.0
        end
    end
    return v
end

function read_image(filename::String, israster::Bool=false)
    israster ? load(filename) : load(filename)
end

end