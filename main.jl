using Plots, Images, FileIO, Statistics, Metalhead, Flux, MLUtils, Augmentor
using Pipe: @pipe

const EPOCHS = 1

include("pipeline.jl")
using .MLImagePipeline

include("models.jl")
using .CatsAndDogsModels

"""Plot 16 randomly selected sample images from the provided image loader."""
function plot_samples(v::LazyImageLoader)
    sample_images = [@pipe sample.filename |> load(_) |> imresize(_, (256, 256)) for sample in rand(v.samples, 16)]
    plot([plot(x, axis=nothing, margin=0Plots.mm) for x in sample_images]..., layout=(4, 4), size=(256*4, 256*4))
    Plots.savefig("samples.png")
end

"Load the sample data into a LazyImageLoader"
function load_data(data_directory::String)
    pipeline = (FlipX() * FlipY() * NoOp()) |> Rotate90(0.25)
    train, test = ClassificationPipeline(data_directory, LabelFromDirectoryName(["cat", "dog"]), pipeline, input_size=(256, 256))
    return DataLoader(train, batchsize=32, shuffle=true), DataLoader(test, batchsize=32, shuffle=false)
end

function accuracy(prediction, ground_truth, labels)
    @pipe (Flux.onecold(prediction, labels) .== Flux.onecold(ground_truth, labels)) |> mean |> x -> x * 100
end

function evaluate(model, dataset)
    predictions = [(model(x), y) for (x, y) in dataset]
    acc = [accuracy(p, y, 0:1) for (p, y) in predictions] |> mean
    loss = [Flux.crossentropy(p, y, dims=1) for (p, y) in predictions] |> mean
    println("Validation Loss: $(round(loss, digits=5, base=10)), Validation Accuracy: $(round(acc, digits=2, base=10))%")
end

function training_callback(accs, losses, prediction, label, iteration, total_iterations, epoch)
    accs = vcat(accs, accuracy(prediction, label, 0:1))
    losses = vcat(losses, Flux.crossentropy(prediction, label, dims=1))
    acc, l = mean(accs), mean(losses)
    println("Epoch $epoch: $iteration / $total_iterations, Loss: $(round(l, digits=4, base=10)), Accuracy: $(round(acc, digits=2, base=10))%")
    return accs, losses
end

"Main program loop"
function main_function()
    # Create Model
    model = pretrained_model() |> gpu

    # Get Parameters
    parameters = Flux.params(model[end-1])

    # Define Loss
    loss(x, y) = Flux.crossentropy(model(x), y, dims=1);

    # Load Data
    data_train, data_test = load_data("data")

    # Define Optimizaer
    opt = Flux.Optimise.ADAM(1e-4)

    # Training Loop
    for epoch in 1:EPOCHS

        accs, losses = [], []
        for (step, (x, y)) in enumerate(data_train)

            # Log Loss And Accuracy
            accs, losses = training_callback(accs, losses, model(x), y, step, length(data_train), epoch)

            # Evaluate On Test Set Every 100 Iterations
            if step % 250 == 0
                evaluate(model, data_test)
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
