using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Collections;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using JetBrains.Annotations;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Shapes;

namespace Gym.Environments.Envs.Atari
{
    // from http://programarcadegames.com/python_examples/show_file.php?file=breakout_simple.py
    public class BreakoutEnv : Env
    {
        private IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;

        private int screenWidth = 400;
        private int screenHeight = 600;

        //properties
        private NumPyRandom random;
        private NDArray state;
        private int _stepsBeyondDone;
        private int _topRowY = 80;
        private int _rowBlockCount = 18;

        // polygons
        private List<Block> Blocks;
        private Ball BallObj;
        private Paddle PaddleObj;

        public BreakoutEnv(IEnvironmentViewerFactoryDelegate viewerFactory)
        {
            _viewerFactory = viewerFactory;
            ActionSpace = new Discrete(3);
            Metadata = new Dict("render.modes", new[] { "human", "rgb_array" }, "video.frames_per_second", 50);

            Seed(0);
        }

        public BreakoutEnv([NotNull] IEnvViewer viewer) : this((IEnvironmentViewerFactoryDelegate)null)
        {
            _viewer = viewer ?? throw new ArgumentNullException(nameof(viewer));

            Seed(0);
        }

        public override NDArray Reset()
        {
            Seed(0);
            //return np.array(state);
            return null; //todo
        }

        public override Step Step(int action)
        {
            PaddleObj.Update(action);
            var done = BallObj.Update();
            var reward = 0;

            var paddlePolygon = PaddleObj.Polygon;
            var ballPolygon = BallObj.Polygon;
            if (paddlePolygon.CollidesWidth(ballPolygon))
            {
                var diff = (paddlePolygon.X + paddlePolygon.Width / 2) - (ballPolygon.X + ballPolygon.Width / 2);

                // Set the ball's y position in case
                // we hit the ball on the edge of the paddle
                BallObj.Y = screenHeight - paddlePolygon.Height - ballPolygon.Height - 1;
                BallObj.Bounce(diff);
            }

            foreach (var block in Blocks.Where(b => !b.Destroyed))
            {
                if (!block.Polygon.CollidesWidth(ballPolygon))
                {
                    continue;
                }

                reward = 1;
                block.Destroyed = true;
                BallObj.Bounce(0);
                break;
            }

            if (Blocks.All(b => b.Destroyed))
            {
                done = true;
            }

            return new Step(np.array(0 /*TODO*/), reward, done, null);
        }

        public override Image<Rgba32> Render(string mode = "human")
        {
            if (_viewer == null)
                lock (this)
                {
                    //to prevent double initalization.
                    if (_viewer == null)
                    {
                        if (_viewerFactory == null)
                            throw new ArgumentNullException(nameof(_viewerFactory), $"No {nameof(_viewerFactory)} have been set");
                        _viewer = _viewerFactory(screenWidth, screenHeight, "breakout");
                    }
                }

            var img = new Image<Rgba32>(screenWidth, screenHeight);

            img.Mutate(i => i.BackgroundColor(Rgba32.Black));
            foreach (var block in Blocks.Where(b => !b.Destroyed))
            {
                img.Mutate(i => i.Fill(block.Color, block.Polygon));
            }

            img.Mutate(i => i.Fill(BallObj.Color, BallObj.Polygon));
            img.Mutate(i => i.Fill(PaddleObj.Color, PaddleObj.Polygon));

            _viewer.Render(img);
            return img;
        }

        /// <remarks>Sets internally stored viewer to null. Might cause problems if factory was not passed.</remarks>
        public override void Close()
        {
            if (_viewer != null)
            {
                _viewer.Close();
                _viewer.Dispose();
                _viewer = null;
            }
        }

        public override void Seed(int seed)
        {
            Blocks = new List<Block>();
            BallObj = new Ball(Rgba32.DarkRed, screenWidth, screenHeight);
            PaddleObj = new Paddle(Rgba32.Red, screenWidth, screenHeight);

            for (var rowIndex = 0; rowIndex < 6; rowIndex++)
            {
                Rgba32 color;
                switch (rowIndex)
                {
                    case 0:
                        color = Rgba32.Red;
                        break;
                    case 1:
                        color = Rgba32.OrangeRed;
                        break;
                    case 2:
                        color = Rgba32.Orange;
                        break;
                    case 3:
                        color = Rgba32.Yellow;
                        break;
                    case 4:
                        color = Rgba32.Green;
                        break;
                    case 5:
                        color = Rgba32.Blue;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                for (var columnIndex = 0; columnIndex < _rowBlockCount; columnIndex++)
                {
                    Blocks.Add(new Block(color, columnIndex * Block.width, rowIndex * Block.height + _topRowY));
                }
            }
        }

        private class Block
        {
            public readonly Rgba32 Color;
            public readonly int X;
            public readonly int Y;
            public bool Destroyed;
            public RectangularPolygon Polygon => new RectangularPolygon(X, Y, width, height);
            public static int width = 23;
            public static int height = 15;

            public Block(Rgba32 color, int x, int y)
            {
                Color = color;
                X = x;
                Y = y;
            }
        }

        private class Ball
        {
            public readonly Rgba32 Color;
            public float X = 0;
            public float Y = 180;
            private readonly int _screenWidth;
            private readonly int _screenHeight;

            public RectangularPolygon Polygon => new RectangularPolygon(X, Y, Width, Height);

            private const int Speed = 10; // Speed in pixels per cycle
            private readonly int Width = 23;
            private readonly int Height = 15;
            private float _direction = 200; //# Direction of ball (in degrees)

            public Ball(Rgba32 color, int screenWidth, int screenHeight)
            {
                Color = color;
                _screenWidth = screenWidth;
                _screenHeight = screenHeight;
            }

            // The 'diff' lets you try to bounce the ball left or right
            // depending where on the paddle you hit it. 0 for walls
            public void Bounce(float diff)
            {
                _direction = (180 - _direction) % 360;
                _direction -= diff;
            }

            public bool Update()
            {
                // Sine and Cosine work in degrees, so we have to convert them
                var direction_radians = ConvertToRadians(_direction);

                // Change the position (x and y) according to the speed and direction
                X += (float)(Speed * Math.Sin(direction_radians));
                Y -= (float)(Speed * Math.Cos(direction_radians));

                // Do we bounce off the top of the screen?
                if (Y <= 0)
                {
                    Bounce(0);
                    Y = 1;
                }

                // Do we bounce of the left side of the screen?
                if (X <= 0)
                {
                    _direction = (360 - _direction) % 360;
                    X = 1;
                }

                // Do we bounce of the right side of the screen?
                if (X > _screenWidth - Width)
                {
                    _direction = (360 - _direction) % 360;
                    X = _screenWidth - Width - 1;
                }

                // Did we fall off the bottom edge of the screen?
                return Y > _screenHeight;
            }

            private double ConvertToRadians(double angle)
            {
                return (Math.PI / 180) * angle;
            }
        }

        private class Paddle
        {
            public readonly Rgba32 Color;
            public RectangularPolygon Polygon => new RectangularPolygon(_x, _y, Width, Height);
            private readonly int _screenWidth;
            private const int Width = 75;
            private const int Height = 15;
            private const int Speed = 15;
            private int _x;
            private readonly int _y;

            public Paddle(Rgba32 color, int screenWidth, int screenHeight)
            {
                Color = color;
                _screenWidth = screenWidth;
                _x = (screenHeight - Width) / 2;
                _y = screenHeight - Height;
            }

            public void Update(int direction)
            {
                switch (direction)
                {
                    case 0:
                        return;
                    case 1:
                        _x -= Speed;
                        if (_x <= 0)
                        {
                            _x = 1;
                        }

                        break;
                    case 2:
                        _x += Speed;
                        if (_x >= _screenWidth - Width - 1)
                        {
                            _x = _screenWidth - Width - 1;
                        }

                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(direction), $"Invalid {nameof(direction)} {direction}. Must be 0 (not moving), 1 (left) or 2 (right)");
                }
            }
        }
    }

    public static class PolygonExtensions
    {
        public static bool CollidesWidth([NotNull] this RectangularPolygon source, [NotNull] RectangularPolygon target)
        {
            if (source == null || target == null)
            {
                throw new ArgumentNullException();
            }

            return source.X < target.X + target.Width &&
                   source.X + source.Width > target.X &&
                   source.Y < target.Y + target.Height &&
                   source.Y + source.Height > target.Y;
        }
    }
}