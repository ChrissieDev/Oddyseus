using System;
using System.Collections.Generic;
using System.Linq;
using Oddyseus.Types;

namespace Oddyseus.Core
{
    // Result of quick text appraisal before smoothing is applied
    public readonly struct Appraisal
    {
        public readonly float ValencePulse;   // raw pulse -1..1
        public readonly float ArousalPulse;   // raw pulse 0..1
        public readonly int Pleasantness;     // -10..10 coarse valence for relationships
        public Appraisal(float v, float a, int p)
        {
            ValencePulse = Math.Clamp(v, -1f, 1f);
            ArousalPulse = Math.Clamp(a, 0f, 1f);
            Pleasantness = Math.Clamp(p, -10, 10);
        }
    }

    public readonly struct EmotionSnapshot
    {
        public readonly float Valence;
        public readonly float Arousal;
        public readonly string Label;
        public readonly DateTime TimestampUtc;
        public EmotionSnapshot(float v, float a, string label)
        {
            Valence = v;
            Arousal = a;
            Label = label;
            TimestampUtc = DateTime.UtcNow;
        }
    }

    public interface IEmotionEngine
    {
        float Valence { get; }
        float Arousal { get; }
        string CurrentLabel { get; }
        Appraisal Appraise(string text);
        void Apply(Appraisal appraisal);
        EmotionSnapshot Snapshot();
        float AffectMatch(float memValence, float memArousal);
    }

    // Minimal emotion engine, it basically keeps only valence/arousal, and derives label on demand
    public sealed class EmotionEngine : IEmotionEngine
    {
        // Current "smoothed" state
        public float Valence { get; private set; }   // -1..1
        public float Arousal { get; private set; }   // 0..1
        public string CurrentLabel => LabelFrom(Valence, Arousal);

        // Smoothing factors. Bigger = bigger reaction
        private const float ValenceBlend = 0.20f;
        private const float ArousalBlend = 0.15f;

        // Neutral drift target (where we slowly glide back to when idle)
        private static readonly (float v, float a) NeutralTarget = (0f, 0.25f);

        // Track last time we ran decay so we can scale decay by real elapsed time
        private DateTimeOffset _lastDecayUtc = DateTimeOffset.MinValue;

        // Heuristic lexicons (small + cheap)
        private static readonly string[] PositiveWords = { "love", "great", "awesome", "nice", "thanks", "thank", "cool", "amazing", "good", "wonderful", "glad", "happy" };
        private static readonly string[] NegativeWords = { "hate", "stupid", "idiot", "dumb", "awful", "terrible", "bad", "angry", "mad", "upset", "annoying", "sad", "sorry" };
        private static readonly string[] CalmingWords  = { "calm", "relax", "breathe", "sleep", "rest", "peace" };
        private static readonly string[] HighArousalMarkers = { "!", "!!!", "now", "hurry", "quick", "urgent" };
        private static readonly string[] LowArousalWords = { "tired", "sleepy", "bored", "boring", "exhausted" };

        // Appraise raw text -> unsmoothed pulses
        public Appraisal Appraise(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Appraisal(0f, 0f, 0);

            var lower = text.ToLowerInvariant();

            int posHits = CountHits(lower, PositiveWords);
            int negHits = CountHits(lower, NegativeWords);

            int net = posHits - negHits;
            // Base valence pulse from word balance
            float valPulse = 0f;
            if (net != 0)
            {
                // Scale: 1 word difference ≈ 0.25, 2+ saturate toward 1
                valPulse = Math.Clamp(net * 0.25f, -1f, 1f);
            }

            // Arousal pulses
            float arousalPulse = 0.25f; // baseline mid-low
            if (HighArousalMarkers.Any(m => lower.Contains(m)))
                arousalPulse += 0.25f;
            if (LowArousalWords.Any(w => lower.Contains(w)))
                arousalPulse -= 0.20f;
            if (CalmingWords.Any(w => lower.Contains(w)))
                arousalPulse -= 0.15f;

            // All caps heuristic (shouting)
            if (HasShouting(text))
                arousalPulse += 0.25f;

            arousalPulse = Math.Clamp(arousalPulse, 0f, 1f);

            // Pleasantness integer for relationship: derive from valPulse * 10
            int pleasantness = (int)Math.Round(valPulse * 10f);

            return new Appraisal(valPulse, arousalPulse, pleasantness);
        }

        public void Apply(Appraisal appraisal)
        {
            // Smooth update
            Valence = Smooth(Valence, appraisal.ValencePulse, ValenceBlend);
            Arousal = Smooth(Arousal, appraisal.ArousalPulse, ArousalBlend);
        }

        public EmotionSnapshot Snapshot() => new EmotionSnapshot(Valence, Arousal, CurrentLabel);

        // We blend a little toward neutral. Call this from a timer or once per turn.
        public void Decay(IClock clock)
        {
            var now = clock.UtcNow;
            if (_lastDecayUtc == DateTimeOffset.MinValue)
            {
                _lastDecayUtc = now;
                return; // no previous baseline yet
            }

            var elapsedSeconds = (now - _lastDecayUtc).TotalSeconds;
            if (elapsedSeconds <= 0) return;

            // How fast we relax per second toward neutral.
            // Example: 0.02 blend per minute, a very gentle drift.
            const double blendPerMinute = 0.02;
            var blend = (float)Math.Min(1.0, (elapsedSeconds / 60.0) * blendPerMinute);

            Valence = Smooth(Valence, NeutralTarget.v, blend);
            Arousal = Smooth(Arousal, NeutralTarget.a, blend);

            _lastDecayUtc = now;
        }

        // Affect match between current state and a stored memory snapshot
        public float AffectMatch(float memValence, float memArousal)
        {
            var dist = AffectDistance(Valence, Arousal, memValence, memArousal);
            // Convert distance to similarity 0..1
            const float roughMax = 1.5f;
            var norm = Math.Clamp(dist / roughMax, 0f, 1f);
            return 1f - norm;
        }

        // Static helpers

        public static float AffectDistance(float v1, float a1, float v2, float a2)
        {
            var dv = v1 - v2;
            var da = (a1 - a2) * 0.8f; // slightly reduce arousal impact
            return MathF.Sqrt(dv * dv + da * da);
        }

        private static float Smooth(float current, float target, float blend)
            => current + (target - current) * blend;

        private static int CountHits(string text, IEnumerable<string> words)
        {
            int c = 0;
            foreach (var w in words)
                if (text.Contains(w))
                    c++;
            return c;
        }

        private static bool HasShouting(string original)
        {
            if (original.Length < 4) return false;
            int upper = 0;
            int letters = 0;
            foreach (var ch in original)
            {
                if (char.IsLetter(ch))
                {
                    letters++;
                    if (char.IsUpper(ch)) upper++;
                }
            }
            return letters > 0 && upper > 0 && (float)upper / letters > 0.6f;
        }

        // Simple label buckets 
        public static string LabelFrom(float v, float a)
        {
            if (v > 0.45f && a > 0.55f) return "Joy";
            if (v > 0.45f && a <= 0.55f) return "Content";
            if (v < -0.45f && a > 0.60f) return "Anger";
            if (v < -0.45f && a <= 0.60f) return "Sad";
            if (Math.Abs(v) < 0.2f && a < 0.25f) return "Calm";
            if (Math.Abs(v) < 0.25f && a > 0.65f) return "Surprised";
            return "Neutral";
        }
    }

    // Extension to stamp a memory with current emotion snapshot
    public static class EmotionMemoryExtensions
    {
        public static void StampEmotion(this MemoryEntry m, IEmotionEngine engine)
        {
            m.Valence = engine.Valence;
            m.Arousal = engine.Arousal;
        }
    }

    // Minimal memory record
    public sealed class MemoryEntry
    {
        public string Id = Guid.NewGuid().ToString();
        public DateTime TimeUtc = DateTime.UtcNow;
        public string Role = "";
        public string UserText = "";
        public string AiText = "";
        public float Valence;          // snapshot
        public float Arousal;          // snapshot
        public int Pleasantness;
        public int RelationshipPoints;
        public float[] Embedding = Array.Empty<float>();

        // Fill label lazily using engine's same logic
        public EmotionSnapshot EmotionSnapshot => new EmotionSnapshot(
            Valence,
            Arousal,
            EmotionEngine.LabelFrom(Valence, Arousal)
        );
    }
}

