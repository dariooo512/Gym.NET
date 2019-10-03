﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AtariDeepQLearner
{
    public class Episode
    {
        public Observation[] Observations { get; set; }
        public float TotalReward => Observations?.Sum(x => x.Reward) ?? throw new InvalidOperationException($"No {nameof(Observations)} set");
    }

    public class ReplayMemory
    {
        public List<Episode> Episodes { get; } = new List<Episode>();
        private List<Observation> Observations { get; } = new List<Observation>();
        private Queue<Image<Rgba32>> _imagesQueue = new Queue<Image<Rgba32>>();
        private int _currentId;
        private readonly int _stageFrames;
        private readonly int _imageWidth;
        private readonly int _imageHeight;

        public ReplayMemory(int stageFrames, int imageWidth, int imageHeight)
        {
            _stageFrames = stageFrames;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
        }

        public void Memorize(Image<Rgba32> prev, int action, float currentReward)
        {
            if (prev.Width != _imageWidth || prev.Height != _imageHeight)
            {
                throw new ArgumentException($"Frame size differs from expected size");
            }

            _imagesQueue.Enqueue(prev);
            if (_imagesQueue.Count <= _stageFrames)
            {
                return;
            }

            _imagesQueue.Dequeue();
            Observations.Add(new Observation(_stageFrames)
            {
                Id = _currentId++,
                ActionTaken = action,
                Reward = currentReward,
                Images = _imagesQueue.ToArray()
            });
        }

        public Image<Rgba32>[] GetCurrent() =>
            Observations.Any() ? Observations.Last().Images.ToArray() : null;

        public void EndEpisode()
        {
            _imagesQueue = new Queue<Image<Rgba32>>();
            Episodes.Add(new Episode { Observations = Observations.ToArray() });
        }

        public void Save(string fileName, int? maxItems = null)
        {
            if (maxItems == null)
            {
                File.WriteAllText(fileName, JsonConvert.SerializeObject(Episodes, Formatting.None));
                return;
            }

            var episodes = Episodes
                .OrderByDescending(x => x.TotalReward)
                .Take(maxItems.Value);

            File.WriteAllText(fileName, JsonConvert.SerializeObject(episodes, Formatting.None));
        }
    }
}