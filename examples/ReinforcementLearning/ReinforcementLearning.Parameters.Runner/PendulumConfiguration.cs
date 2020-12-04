using System;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.Avalonia;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning.Parameters.Runner {
    public sealed class PendulumConfiguration : IParametersGameConfiguration {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new PendulumEnv(AvaloniaEnvViewer.Run));

        public IEnv EnvInstance => _env.Value;
        public int MemoryStates => 1;
        public int MemoryCapacity => 25;
        public int SkippedFrames => 1;
        public int ParametersLength => 3; // cos, sin, acceleration
        public float StartingEpsilon => 1F;
        public int Episodes => 400;
        public int BatchSize => 100;
        public int Epochs => 10;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
            NetworkLayers.FullyConnected(20, ActivationType.Sigmoid),
            NetworkLayers.FullyConnected(10, ActivationType.Sigmoid),
            NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));

        public INeuralNetwork BuildCudaNeuralNetwork()=>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
            CuDnnNetworkLayers.FullyConnected(20, ActivationType.Sigmoid),
            CuDnnNetworkLayers.FullyConnected(10, ActivationType.Sigmoid),
            CuDnnNetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}