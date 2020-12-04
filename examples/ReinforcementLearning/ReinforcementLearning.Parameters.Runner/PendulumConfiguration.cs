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
        public int SkippedFrames => 0;
        public int ParametersLength => 2; // newth, newthdot
        public float StartingEpsilon => .9F;
        public int Episodes => 500;
        public int BatchSize => 100;
        public int Epochs => 15;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
            NetworkLayers.FullyConnected(20, ActivationType.ReLU),
            NetworkLayers.FullyConnected(5, ActivationType.ReLU),
            NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));

        public INeuralNetwork BuildCudaNeuralNetwork()=>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
            CuDnnNetworkLayers.FullyConnected(20, ActivationType.ReLU),
            CuDnnNetworkLayers.FullyConnected(5, ActivationType.ReLU),
            CuDnnNetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}