using System;
using Gym.Environments.Envs.Atari;
using Gym.Envs;
using Gym.Rendering.WinForm.Rendering;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.Images.Runner {
    public sealed class BreakoutConfiguration : IImageGameConfiguration {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new BreakoutEnv(WinFormViewer.Run));

        public IEnv EnvInstance => _env.Value;
        public int MemoryStates => 2;
        public ImageStackLayout ImageStackLayout => ImageStackLayout.Vertical;
        public int MemoryCapacity => 10;
        public int SkippedFrames => 3;
        public int FrameWidth => 400;
        public int FrameHeight => 600;
        public int ScaledImageWidth => 40;
        public int ScaledImageHeight => 40;
        public FramePadding FramePadding => null;
        public float StartingEpsilon => 1F;
        public int Episodes => 2000;
        public int BatchSize => 100;
        public int Epochs => 10;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(ScaledImageHeight, ScaledImageWidth),
                NetworkLayers.Convolutional((10, 10), 40, ActivationType.ReLU),
                NetworkLayers.FullyConnected(20, ActivationType.ReLU),
                NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}