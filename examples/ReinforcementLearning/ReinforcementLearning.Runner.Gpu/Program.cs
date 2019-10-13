namespace ReinforcementLearning.Runner.Gpu
{
    class Program
    {
        static void Main(string[] args)
        {
            new GameEngine().Play(new CartPoleConfiguration());
        }
    }
}
