using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using Gym.Collections;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using JetBrains.Annotations;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;
using SixLabors.Shapes;

namespace Gym.Environments.Envs.Classic {
    public class PendulumEnv : Env {
        private IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;

        //constants
        private const float max_speed = 8;
        private const float max_torque = 2.0f;
        private const float dt = 0.05f;
        private const float g = 10.0f;
        private const float m = 1;
        private const float l = 1;
        private const float episode_steps = 100;
        private float current_step = 0;

        //properties
        private NumPyRandom random;
        private NDArray state;
        private int last_u;

        public PendulumEnv(IEnvironmentViewerFactoryDelegate viewerFactory, NumPyRandom randomState) {
            _viewerFactory = viewerFactory;
            var high = np.array(1f, 1f, max_speed);
            ActionSpace = new Discrete(2);
            ObservationSpace = new Box(-high, high, np.float32);
            random = randomState ?? np.random.RandomState();

            Metadata = new Dict("render.modes", new[] {"human", "rgb_array"}, "video.frames_per_second", 30);
        }

        public PendulumEnv(IEnvironmentViewerFactoryDelegate viewerFactory) : this(viewerFactory, null) { }

        public PendulumEnv([NotNull] IEnvViewer viewer) : this((IEnvironmentViewerFactoryDelegate) null) {
            _viewer = viewer ?? throw new ArgumentNullException(nameof(viewer));
        }

        public override NDArray Reset() {
            current_step = 0;
            var high = np.array(np.pi, 1);
            state = random.uniform(-high, high, np.@double);
            return np.array(state);
        }

        public override Step Step(int action) {
            current_step++;
            Debug.Assert(ActionSpace.Contains(action), $"{action} ({action.GetType().Name}) invalid action for {GetType().Name} environment");
            var th = state.GetDouble();
            var thdot = state.GetDouble(1);


            var force = action == 1 ? max_torque : -max_torque;
            // var force = max_torque*5;
            // var force = 0;

            last_u = action; // for rendering
            var reward = -(float) (Math.Pow(angle_normalize(th), 2) + .1 * Math.Pow(thdot, 2) + .001 * (Math.Pow(force, 2)));

            double newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3 / (m * Math.Pow(l, 2)) * force) * dt;
            double newth = th + newthdot * dt;
            newthdot = np.clip(newthdot, -max_speed, max_speed);

            state = np.array(newth, newthdot);
            var aaaaaa = get_obs(state);

            // reward = (float)aaaaaa.GetDouble(0);
            
            return new Step(get_obs(state), reward, current_step > episode_steps, null);
        }
        
        private NDArray get_obs(NDArray ndArray) {
            return np.array(new double[] {np.cos(ndArray.GetDouble(0)), np.sin(ndArray.GetDouble(0)), ndArray.GetDouble(1)});
        }

        readonly Image clockwiseImage = Image.Load("./Envs/Classic/assets/clockwise.png");

        public override Image Render(string mode = "human") {
            float b, t, r, l;
            const int screen_width = 500;
            const int screen_height = 500;
            const int center_x = screen_width / 2;
            const int center_y = screen_height / 2;
            const int clockwiseImage_x = 50;
            const int clockwiseImage_y = 50;
            var center = new PointF(center_x, center_y);


            if (_viewer == null) {
                lock (this) {
                    //to prevent double initalization.
                    if (_viewer == null) {
                        if (_viewerFactory == null)
                            _viewerFactory = NullEnvViewer.Factory;
                        _viewer = _viewerFactory(screen_width, screen_height, "Pendulum-v0");
                    }
                }
            }

            var draw = new List<(IPath, Rgba32)>();
            var rodStart = new EllipsePolygon(center_x, center_y, 20, 20);
            var rodEnd = new EllipsePolygon(center_x - 100, center_y, 20, 20);
            var rodStick = new RectangularPolygon(center_x, center_y - 10, -100, 20);
            var rod = new ComplexPolygon(rodStart, rodEnd, rodStick);
            draw.Add((rod.Transform(Matrix3x2.CreateRotation((float) ((float) state.GetDouble(0) + np.pi / 2), center)), new Rgba32(211, 110, 109)));
            draw.Add((new EllipsePolygon(center_x, center_y, 10, 10), Rgba32.Black));

            clockwiseImage.Mutate(x => {
                x.Resize(clockwiseImage_x, clockwiseImage_y);
                if (last_u ==1) {
                    x.Flip(FlipMode.Horizontal);
                }
            });

            var img = new Image<Rgba32>(screen_width, screen_height);
            img.Mutate(i => i.DrawImage(clockwiseImage, new Point(center_x - clockwiseImage_x / 2, center_y - clockwiseImage_y / 2), 1));

            //line
            img.Mutate(i => i.BackgroundColor(Rgba32.White));
            foreach (var (path, rgba32) in draw) {
                img.Mutate(i => i.Fill(rgba32, path));
            }

            _viewer.Render(img);
            return img;
        }


        public override void Close() {
            if (_viewer != null) {
                _viewer.Close();
                _viewer.Dispose();
                _viewer = null;
            }
        }
        public override void Seed(int seed) {
            random = np.random.RandomState(seed);
        }

        private double angle_normalize(double x) {
            return (((x + np.pi) % (2 * np.pi)) - np.pi);
        }
    }
}