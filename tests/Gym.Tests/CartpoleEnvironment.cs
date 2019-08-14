﻿using System;
using System.Threading;
using Ebby.Gym.Envs.Classic;
using Gym.Tests.Helpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests {
    [TestClass]
    public class CartpoleEnvironment {
        [TestMethod]
        public void Run() {
            var cp = new CartPoleEnv();
            var rnd = new Random();
            var done = true;
            using (new StopwatchMeasurer("time it took to run all steps in ms"))
                for (int i = 0; i < 100_000; i++) {
                    if (done) {
                        cp.Reset();
                        done = false;
                    } else {
                        var (observation, reward, _done, information) = cp.Step((i % 2));
                        done = _done;
                    }

                    cp.Render();
                    Thread.Sleep(15); //this is to prevent it from finishing instantly !
                }
        }
    }
}
