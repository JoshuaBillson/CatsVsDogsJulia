module MLImagePipeline
export LabelFromDirectoryName, LabelFromFilename, MaskFromID, AbstractLabel, ClassificationPipeline

using Plots, Images, FileIO, Statistics, Metalhead, Flux, MLUtils, Random
using Pipe: @pipe
using Augmentor: Pipeline, augment, Resize, NoOp, augmentbatch!

include("utils.jl")
using .Utils

abstract type AbstractLabel end

abstract type AbstractMask <: AbstractLabel end

abstract type AbstractOneHot <: AbstractLabel end

"Callable struct for extracting a one-hot label based on the subdirectory in which a sample image is located"
struct LabelFromDirectoryName <: AbstractOneHot
    classes::Vector{String}
end

"Callable which takes the path to a sample image and returns its label as a one-hot encoded vector based on the subdirectory name"
(l::LabelFromDirectoryName)(path) = @pipe split(path, "/")[end-1] |> one_hot(_, l.classes)

struct LabelFromFilename <: AbstractOneHot
    classes::Vector{String}
end

(l::LabelFromFilename)(path) = @pipe split(path, "/")[end] |> filter(class -> contains(_, class), l.classes)[1] |> one_hot(_, l.classes)

struct MaskFromID <: AbstractMask
    masks::Dict{Int, String}
    read_mask::Function
end

(m::MaskFromID)(id::Int) = m.masks[id] |> m.read_mask

"A struct which lazily reads sample images into memory on-demand"
struct ClassificationPipeline{T <: AbstractLabel}
    samples::Vector{String}
    labels::T
    pipeline::Pipeline
    israster::Bool
end

"A struct which lazily reads sample images into memory on-demand"
struct SegmentationPipeline
    samples::Vector{Int}
    masks::MaskFromID
    pipeline::Pipeline
    israster::Bool
end

function ClassificationPipeline(samples::Vector{String}, l::AbstractLabel, pipeline::Pipeline; at=0.8::Float64, input_size=nothing, israster::Bool=false)
    train, test = @pipe splitobs(samples, at=at) |> x -> (collect(x[1]), collect(x[2]))
    resize = isnothing(input_size) ? NoOp() : Resize(input_size...)
    return ClassificationPipeline(train, l, pipeline |> resize, israster), ClassificationPipeline(test, l, Pipeline(resize), israster)
end

function ClassificationPipeline(data_directory::String, l::LabelFromDirectoryName, pipeline::Pipeline; at=0.8::Float64, input_size=nothing, israster::Bool=false)
    subdirs = ["$data_directory/$class" for class in l.classes]
    samples = ["$subdir/$filename" for subdir in subdirs for filename in readdir(subdir)] |> Random.shuffle
    return ClassificationPipeline(samples, l, pipeline, at=at, input_size=input_size, israster=israster)
end

function ClassificationPipeline(data_directory::String, l::LabelFromFilename, pipeline::Pipeline; at=0.8::Float64, input_size=nothing, israster::Bool=false)
    samples = ["$data_directory/$filename" for filename in readdir(data_directory)]
    return ClassificationPipeline(samples, l, pipeline, at=at, input_size=input_size, israster=israster)
end

function SegmentationPipeline(data_directory::String, subdirs::Vector{String}, ids::AbstractArray{Int}, id_to_filename::Function)
    "Hello World!"
end

"Implement Base.length for LazyImageLoader"
function Base.length(v::ClassificationPipeline)::Integer 
    return length(v.samples)
end

"Implement Base.getindex for LazyImageLoader when i is an int"
function Base.getindex(X::ClassificationPipeline, i::Int)
    filename = X.samples[i]
    x = @pipe load(filename) |> augment(_, X.pipeline) |> float32 |> channelview |>  permutedims(_, (3, 2, 1)) |> reshape(_, (size(_)..., 1))
    y = X.labels(filename)
    return x |> gpu, y |> gpu
end

"Implement Base.getindex for LazyImageLoader when i is a range"
function Base.getindex(X::ClassificationPipeline, i::AbstractArray)
    xs = [@pipe load(X.samples[j]) |> augment(_, X.pipeline) for j in i]
    xs = [@pipe x |> float32 |> channelview |>  permutedims(_, (3, 2, 1)) |> reshape(_, (size(_)..., 1)) for x in xs]
    xs = cat(xs..., dims=(4))
    ys = hcat([X.labels(X.samples[j]) for j in i]...)
    return xs |> gpu, ys |> gpu
end

"Implement Base.getindex for LazyImageLoader when i is an int"
function plot_sample(X::ClassificationPipeline, i::Int)
    filename = X.samples[i]
    features = @pipe load(filename) |> augment(_, X.pipeline)
    p = plot(features, axis=nothing, showaxis=false, margin=0Plots.mm, size=size(features));
    return p
end

end