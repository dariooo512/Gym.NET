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
        public int MemoryStates => 2;
        public int MemoryCapacity => 100;
        public int SkippedFrames => 2;
        public int ParametersLength => 3;
        public float StartingEpsilon => 1F;
        public int Episodes => 2000;
        public int BatchSize => 100;
        public int Epochs => 10;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
                NetworkLayers.FullyConnected(20, ActivationType.ReLU),
                NetworkLayers.FullyConnected(5, ActivationType.ReLU),
                NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));

        public INeuralNetwork BuildCudaNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
                CuDnnNetworkLayers.FullyConnected(50, ActivationType.ReLU),
                CuDnnNetworkLayers.FullyConnected(20, ActivationType.ReLU),
                CuDnnNetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}