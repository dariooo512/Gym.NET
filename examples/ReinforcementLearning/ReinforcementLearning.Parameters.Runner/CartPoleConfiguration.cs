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
    public sealed class CartPoleConfiguration : IParametersGameConfiguration {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new CartPoleEnv(AvaloniaEnvViewer.Run));

        public IEnv EnvInstance => _env.Value;
        public int MemoryStates => 4;
        public int MemoryCapacity => 50;
        public int SkippedFrames => 0;
        public int ParametersLength => 4; // x, x_dot, theta, theta_dot
        public float StartingEpsilon => .7F;
        public int Episodes => 2000;
        public int BatchSize => 100;
        public int Epochs => 5;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
                NetworkLayers.FullyConnected(10, ActivationType.ReLU),
                NetworkLayers.FullyConnected(5, ActivationType.ReLU),
                NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}