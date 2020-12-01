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
        private readonly IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;

        private const int ScreenWidth = 400;
        private const int ScreenHeight = 600;
        private const int TopRowY = 80;
        private const int RowBlockCount = 18;

        // polygons
        private List<Block> _blocks;
        private Ball _ballObj;
        private Paddle _paddleObj;

        public BreakoutEnv(IEnvironmentViewerFactoryDelegate viewerFactory)
        {
            _viewerFactory = viewerFactory;
            ActionSpace = new Discrete(3);
            Metadata = new Dict("render.modes", new[] {"human", "rgb_array"}, "video.frames_per_second", 50);

            Seed(0);
        }

        public BreakoutEnv([NotNull] IEnvViewer viewer) : this((IEnvironmentViewerFactoryDelegate) null)
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
            _paddleObj.Update(action);
            var done = _ballObj.Update();
            var reward = 0;

            var paddlePolygon = _paddleObj.Polygon;
            var ballPolygon = _ballObj.Polygon;
            if (paddlePolygon.CollidesWidth(ballPolygon))
            {
                var diff = (paddlePolygon.X + paddlePolygon.Width / 2) - (ballPolygon.X + ballPolygon.Width / 2);

                // Set the ball's y position in case
                // we hit the ball on the edge of the paddle
                _ballObj.Y = ScreenHeight - paddlePolygon.Height - ballPolygon.Height - 1;
                _ballObj.Bounce(diff);
            }

            foreach (var block in _blocks.Where(b => !b.Destroyed))
            {
                if (!block.Polygon.CollidesWidth(ballPolygon))
                {
                    continue;
                }

                reward = 1;
                block.Destroyed = true;
                _ballObj.Bounce(0);
                break;
            }

            if (_blocks.All(b => b.Destroyed))
            {
                done = true;
            }

            var paddleXPosition = paddlePolygon.X / (ScreenWidth - paddlePolygon.Width);
            var ballXPosition = _ballObj.X / (ScreenWidth - _ballObj.Polygon.Width);
            var ballYPosition = _ballObj.Y / (ScreenHeight - _ballObj.Polygon.Height);
            var ballDirection = _ballObj.Direction / 360;
            
            return new Step(np.array(paddleXPosition, ballXPosition, ballYPosition, ballDirection), reward, done, null);
        }

        public override Image Render(string mode = "human")
        {
            if (_viewer == null)
                lock (this)
                {
                    //to prevent double initalization.
                    if (_viewer == null)
                    {
                        if (_viewerFactory == null)
                            throw new ArgumentNullException(nameof(_viewerFactory), $"No {nameof(_viewerFactory)} have been set");
                        _viewer = _viewerFactory(ScreenWidth, ScreenHeight, "breakout");
                    }
                }

            var img = new Image<Rgba32>(ScreenWidth, ScreenHeight);

            img.Mutate(i => i.BackgroundColor(Rgba32.Black));
            foreach (var block in _blocks.Where(b => !b.Destroyed))
            {
                img.Mutate(i => i.Fill(block.Color, block.Polygon));
            }

            img.Mutate(i => i.Fill(_ballObj.Color, _ballObj.Polygon));
            img.Mutate(i => i.Fill(_paddleObj.Color, _paddleObj.Polygon));

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
            _blocks = new List<Block>();
            _ballObj = new Ball(Rgba32.DarkRed, ScreenWidth, ScreenHeight);
            _paddleObj = new Paddle(Rgba32.Red, ScreenWidth, ScreenHeight);

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

                for (var columnIndex = 0; columnIndex < RowBlockCount; columnIndex++)
                {
                    _blocks.Add(new Block(color, columnIndex * Block.width, rowIndex * Block.height + TopRowY));
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
            public float Direction = 200; // Direction of ball (in degrees)
            private readonly int _screenWidth;
            private readonly int _screenHeight;

            public RectangularPolygon Polygon => new RectangularPolygon(X, Y, Width, Height);

            private const int Speed = 10; // Speed in pixels per cycle
            private readonly int Width = 8;
            private readonly int Height = 15;

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
                Direction = (180 - Direction) % 360;
                Direction -= diff;

                // avoiding super flat angles on horizontal axis
                const int minAngle = 10;
                if (Direction < 90 && Direction - 90 > -minAngle)
                {
                    Direction = 90 - minAngle;
                    return;
                }

                if (Direction > 90 && Direction - 90 < minAngle)
                {
                    Direction = 90 + minAngle;
                    return;
                }

                if (Direction < 270 && Direction - 270 > -minAngle)
                {
                    Direction = 270 - minAngle;
                    return;
                }

                if (Direction > 270 && Direction - 270 < minAngle)
                {
                    Direction = 270 + minAngle;
                }
            }

            public bool Update()
            {
                // Sine and Cosine work in degrees, so we have to convert them
                var direction_radians = ConvertToRadians(Direction);

                // Change the position (x and y) according to the speed and direction
                X += (float) (Speed * Math.Sin(direction_radians));
                Y -= (float) (Speed * Math.Cos(direction_radians));

                // Do we bounce off the top of the screen?
                if (Y <= 0)
                {
                    Bounce(0);
                    Y = 1;
                }

                // Do we bounce of the left side of the screen?
                if (X <= 0)
                {
                    Direction = (360 - Direction) % 360;
                    X = 1;
                }

                // Do we bounce of the right side of the screen?
                if (X > _screenWidth - Width)
                {
                    Direction = (360 - Direction) % 360;
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
                _x = (screenWidth - Width) / 2;
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