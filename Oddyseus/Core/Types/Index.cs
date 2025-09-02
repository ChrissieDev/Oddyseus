using System;
using System.Collections.Generic;
using System.Linq;

namespace Oddyseus.Types
{
    // Emotion labeling + vector representation 
    public enum EmotionalLabel
    {
        Neutral, Joy, Content, Love, Curiosity, Surprise, Pride, Calm, Anticipation,
        Fear, Anxiety, Sadness, Disgust, Anger, Frustration, Shame, Guilt, Boredom, Relief, Determination
    }
    

    public readonly struct EmotionVector
    {
        public float Valence { get; } // -1 .. +1
        public float Arousal { get; } // 0 .. 1
        public EmotionVector(float valence, float arousal)
        {
            Valence = Math.Clamp(valence, -1f, 1f);
            Arousal = Math.Clamp(arousal, 0f, 1f);
        }
        public EmotionVector Lerp(in EmotionVector other, float t) => new(
            Valence + (other.Valence - Valence) * t,
            Arousal + (other.Arousal - Arousal) * t
        );
        public float Distance(in EmotionVector other, float valenceWeight = 1f, float arousalWeight = 0.8f)
        {
            var dv = (Valence - other.Valence) * valenceWeight;
            var da = (Arousal - other.Arousal) * arousalWeight;
            return MathF.Sqrt(dv * dv + da * da);
        }
        public override string ToString() => $"(v={Valence:0.00}, a={Arousal:0.00})";
    }

    public static class EmotionCentroids
    {
        private static readonly Dictionary<EmotionalLabel, EmotionVector> Map = new()
        {
            { EmotionalLabel.Joy,            new EmotionVector( 0.80f, 0.70f) },
            { EmotionalLabel.Content,        new EmotionVector( 0.50f, 0.30f) },
            { EmotionalLabel.Love,           new EmotionVector( 0.70f, 0.50f) },
            { EmotionalLabel.Curiosity,      new EmotionVector( 0.30f, 0.60f) },
            { EmotionalLabel.Surprise,       new EmotionVector( 0.10f, 0.90f) },
            { EmotionalLabel.Pride,          new EmotionVector( 0.60f, 0.60f) },
            { EmotionalLabel.Calm,           new EmotionVector( 0.20f, 0.15f) },
            { EmotionalLabel.Neutral,        new EmotionVector( 0.00f, 0.20f) },
            { EmotionalLabel.Anticipation,   new EmotionVector( 0.20f, 0.55f) },
            { EmotionalLabel.Fear,           new EmotionVector(-0.70f, 0.85f) },
            { EmotionalLabel.Anxiety,        new EmotionVector(-0.60f, 0.75f) },
            { EmotionalLabel.Sadness,        new EmotionVector(-0.70f, 0.30f) },
            { EmotionalLabel.Disgust,        new EmotionVector(-0.60f, 0.40f) },
            { EmotionalLabel.Anger,          new EmotionVector(-0.75f, 0.80f) },
            { EmotionalLabel.Frustration,    new EmotionVector(-0.50f, 0.65f) },
            { EmotionalLabel.Shame,          new EmotionVector(-0.65f, 0.35f) },
            { EmotionalLabel.Guilt,          new EmotionVector(-0.55f, 0.45f) },
            { EmotionalLabel.Boredom,        new EmotionVector(-0.20f, 0.10f) },
            { EmotionalLabel.Relief,         new EmotionVector( 0.30f, 0.25f) },
            { EmotionalLabel.Determination,  new EmotionVector( 0.40f, 0.55f) }
        };

        public static EmotionVector Get(EmotionalLabel label) => Map[label];
        public static EmotionalLabel Nearest(in EmotionVector v, out float distance)
        {
            EmotionalLabel best = EmotionalLabel.Neutral;
            distance = float.MaxValue;
            foreach (var kv in Map)
            {
                var d = kv.Value.Distance(v);
                if (d < distance)
                {
                    distance = d;
                    best = kv.Key;
                }
            }
            return best;
        }
        public static IReadOnlyDictionary<EmotionalLabel, EmotionVector> All => Map;
    }

    public sealed class EmotionState
    {
        public EmotionVector Vector { get; private set; } = EmotionCentroids.Get(EmotionalLabel.Neutral);
        public EmotionalLabel Label { get; private set; } = EmotionalLabel.Neutral;
        private const float SmoothFactor = 0.25f;
        private const float RelabelThreshold = 0.15f;
        public void Update(EmotionVector incoming)
        {
            Vector = Vector.Lerp(incoming, SmoothFactor);
            var current = EmotionCentroids.Get(Label);
            if (current.Distance(Vector) > RelabelThreshold)
            {
                Label = EmotionCentroids.Nearest(Vector, out _);
            }
        }
        public void Decay(float drift = 0.02f)
        {
            var neutral = EmotionCentroids.Get(EmotionalLabel.Neutral);
            Vector = Vector.Lerp(neutral, drift);
        }
    }

    public class RelationshipData
    {
        public string? UserName { get; set; }
        public int InteractionCount { get; set; }
        public int RelationshipPoints { get; set; }  // -100..100
        public float AffinityScore { get; set; }
        public void UpdateRelationshipPoints(int delta) => RelationshipPoints = Math.Clamp(RelationshipPoints + delta, -100, 100);
        public string Partition()
        {
            var p = RelationshipPoints;
            if (p <= -80) return "Hatred";
            if (p <= -50) return "Disgust";
            if (p <= -35) return "StrongDislike";
            if (p <= -15) return "Dislike";
            if (p <= -5)  return "MildDislike";
            if (p < 0)    return "Uneasy";
            if (p == 0)   return "Neutral";
            if (p <= 15)  return "MildLike";
            if (p <= 35)  return "Like";
            if (p <= 55)  return "StrongLike";
            if (p <= 75)  return "Admiration";
            if (p <= 90)  return "Affection";
            return "Love";
        }
    }

    public class MemoryEntry
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string Role { get; set; } = string.Empty; // user / ai
        public string Text { get; set; } = string.Empty;
        public EmotionalLabel? EmotionLabel { get; set; }
        public EmotionVector? EmotionVector { get; set; }
        public int Pleasantness { get; set; }
    }

    public interface IEmotionalStateManager
    {
        EmotionVector CurrentVector { get; }
        EmotionalLabel CurrentLabel { get; }
        void Apply(EmotionVector v);
        void Decay();
    }

    public sealed class EmotionalStateManager : IEmotionalStateManager
    {
        private readonly EmotionState _state = new();
        public EmotionVector CurrentVector => _state.Vector;
        public EmotionalLabel CurrentLabel => _state.Label;
        public void Apply(EmotionVector v) => _state.Update(v);
        public void Decay() => _state.Decay();
    }

    public interface IRelationshipModeler
    {
        void AddInteraction(string user);
        RelationshipData GetRelationshipData(string user);
        void AdjustPoints(string user, int delta);
    }
}
