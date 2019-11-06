using System;
using Gym.Environments.Envs.Atari;
using Gym.Envs;
using Gym.Rendering.WinForm;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.Images.Runner {
    public sealed class BreakoutConfiguration : IImageGameConfiguration {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new BreakoutEnv(WinFormEnvViewer.Run));

        public IEnv EnvInstance => _env.Value;
        public int MemoryStates => 2;
        public ImageStackLayout ImageStackLayout => ImageStackLayout.Horizontal;
        public int MemoryCapacity => 10;
        public int SkippedFrames => 3;
        public int FrameWidth => 400;
        public int FrameHeight => 600;
        public int ScaledImageWidth => 60;
        public int ScaledImageHeight => 60;
        public FramePadding FramePadding => null;
        public float StartingEpsilon => 1F;
        public int Episodes => 500;
        public int BatchSize => 50;
        public int Epochs => 5;

        public INeuralNetwork BuildNeuralNetwork() => NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(ScaledImageHeight, ScaledImageWidth),
            NetworkLayers.Convolutional((2, 2), 40, ActivationType.ReLU),
            NetworkLayers.FullyConnected(20, ActivationType.ReLU),
            NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));

        public INeuralNetwork BuildCudaNeuralNetwork() => NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(ScaledImageHeight, ScaledImageWidth),
            CuDnnNetworkLayers.Convolutional((2, 2), 40, ActivationType.ReLU),
            CuDnnNetworkLayers.FullyConnected(20, ActivationType.ReLU),
            CuDnnNetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}