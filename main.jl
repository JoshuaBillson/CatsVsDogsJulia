using Plots, Images, FileIO, Statistics, Metalhead, Flux, MLUtils, Augmentor, Metrics, ProgressBars
using Pipe: @pipe

const EPOCHS = 1

include("pipeline.jl")
using .MLImagePipeline

include("models.jl")
using .CatsAndDogsModels

include("utils.jl")
using .Utils

"""Plot 16 randomly selected sample images from the provided image loader."""
function plot_samples(v::ClassificationPipeline, filename::String)
    plot([MLImagePipeline.plot_sample(v, i) for i in rand(1:length(v), 16)]..., layout=(4, 4), size=(256*4, 256*4))
    Plots.savefig(filename)
end

"Load the sample data into a LazyImageLoader"
function load_data(data_directory::String)
    pipeline = (FlipX() * FlipY() * NoOp()) |> Rotate90(0.25)
    train, test = ClassificationPipeline(data_directory, LabelFromDirectoryName(["cat", "dog"]), pipeline, input_size=(256, 256))
    plot_samples(train, "images/train.png")
    plot_samples(test, "images/test.png")
    return DataLoader(train, batchsize=32, shuffle=true), DataLoader(test, batchsize=32, shuffle=false)
end

function training_callback(prediction, label, aggregator::Utils.MetricAggregator, iteration, total_iterations, epoch)
    evaluation = aggregator(prediction, label)
    println("Epoch $epoch: $iteration / $total_iterations, $evaluation")
end

function evaluate_model(model, data, iter)
    predictions = [(Array(model(x)), Array(y)) for (x, y) in data]
    accuracy = [Metrics.categorical_accuracy(p, l) for (p, l) in predictions] |> mean
    loss = [Flux.Losses.crossentropy(p, l) for (p, l) in predictions] |> mean
    println(iter, "Test Loss: $(round(loss, digits=6, base=10)), Test Accuracy: $(round(accuracy * 100, digits=4, base=10))%")
end

"Main program loop"
function main_function()
    # Create Model
    model = Flux.Chain(Metalhead.ResNet(34, pretrain=true), Flux.Dense(1000, 2), Flux.softmax) |> gpu

    # Get Parameters
    parameters = Flux.params(model[end-1])

    # Define Loss
    loss(x, y) = Flux.crossentropy(model(x), y, dims=1);

    # Load Data
    data_train, data_test = load_data("data")

    # Define Optimizaer
    opt = Flux.Optimise.ADAM(1e-4)

    # Define Metrics
    loss_accumulator = Utils.MetricAccumulator(Flux.Losses.crossentropy, m -> "Loss: $(round(m, digits=6, base=10))")
    accuracy_accumulator = Utils.MetricAccumulator(Metrics.categorical_accuracy, m -> "Accuracy: $(round(m*100.0, digits=4, base=10))%")
    aggregator =  Utils.MetricAggregator([loss_accumulator, accuracy_accumulator])

    # Training Loop
    for epoch in 1:EPOCHS

        iter = ProgressBar(data_train)
        for (step, (x, y)) in enumerate(iter)

            # Log Loss And Accuracj
            set_description(iter, aggregator(Array(model(x)), Array(y)))

            # Evaluate On Test Set Every 250 Iterations
            if step % 250 == 0
                evaluate_model(model, data_test, iter)
            end

            # Compute Gradients
            grads = Flux.gradient(() -> loss(x, y), parameters)

            # Update Parameters
            Flux.Optimise.update!(opt, parameters, grads)
        end
    end
end

"Program entry point"
main = main_function()
