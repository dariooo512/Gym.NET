using System;
using Gym.Environments.Envs.Atari;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.Avalonia;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning.Parameters.Runner
{
    public sealed class BreakoutConfiguration : IParametersGameConfiguration
    {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new BreakoutEnv(AvaloniaEnvViewer.Run));

        public IEnv EnvInstance => _env.Value;
        public int MemoryStates => 1;
        public int MemoryCapacity => 25;
        public int SkippedFrames => 0;
        public int ParametersLength => 4; // paddleXPosition, ballXPosition, ballYPosition, ballDirection
        public float StartingEpsilon => .9F;
        public int Episodes => 1000;
        public int BatchSize => 50;
        public int Epochs => 6;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
                NetworkLayers.FullyConnected(10, ActivationType.LeakyReLU),
                NetworkLayers.FullyConnected(10, ActivationType.Sigmoid),
                NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));

        public INeuralNetwork BuildCudaNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
                CuDnnNetworkLayers.FullyConnected(10, ActivationType.LeakyReLU),
                CuDnnNetworkLayers.FullyConnected(10, ActivationType.Sigmoid),
                CuDnnNetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}