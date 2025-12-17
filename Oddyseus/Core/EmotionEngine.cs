using System;
using System.Collections.Generic;
using Oddyseus.Types;

namespace Oddyseus.Core
{
    public readonly record struct RelationshipSnapshot(string UserId, int Points, string Label, float Affinity)
    {
        public static RelationshipSnapshot Empty => new("", 0, "Neutral", 0f);
    }

    public readonly record struct AppraisalContext(
        string Text,
        EmotionSnapshot CurrentEmotion,
        RelationshipSnapshot Relationship,
        IReadOnlyList<MemoryEntry> CandidateMemories);

    public interface IAppraisalProvider
    {
        Appraisal Evaluate(AppraisalContext context);
    }

    public sealed class NeutralAppraisalProvider : IAppraisalProvider
    {
        public Appraisal Evaluate(AppraisalContext context) => new Appraisal(0f, 0f, 0);
    }

    // quick pulse read before the smoothing pass kicks in
    public readonly struct Appraisal
    {
        public readonly float ValencePulse;   // -1..1 snapshot
        public readonly float ArousalPulse;   // 0..1 snapshot
        public readonly int Pleasantness;     // -10..10 coarse vibe for relationship math
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
        Appraisal Appraise(string text, RelationshipSnapshot relationship, IReadOnlyList<MemoryEntry>? candidateMemories = null);
        void Apply(Appraisal appraisal);
        EmotionSnapshot Snapshot();
        float AffectMatch(float memValence, float memArousal);
    }

    // dead-simple engine: glide valence/arousal around, slap a label on demand
    public sealed class EmotionEngine : IEmotionEngine
    {
        private readonly IAppraisalProvider _appraisalProvider;

        public EmotionEngine(IAppraisalProvider? appraisalProvider = null)
        {
            _appraisalProvider = appraisalProvider ?? new NeutralAppraisalProvider();
        }

        public float Valence { get; private set; }   // -1..1 running state
        public float Arousal { get; private set; }   // 0..1 running state
        public string CurrentLabel => LabelFrom(Valence, Arousal);

        // bigger blend = snappier mood swings
        private const float ValenceBlend = 0.20f;
        private const float ArousalBlend = 0.15f;

        // neutral drift anchors when nothing exciting happens
        private static readonly (float v, float a) NeutralTarget = (0f, 0.25f);

        // keep last decay tick so time-based drift feels real
        private DateTimeOffset _lastDecayUtc = DateTimeOffset.MinValue;

        // dumb heuristics until the lightweight LLM takes over
        private static readonly string[] PositiveWords = { "love", "great", "awesome", "nice", "thanks", "thank", "cool", "amazing", "good", "wonderful", "glad", "happy" };
        private static readonly string[] NegativeWords = { "hate", "stupid", "idiot", "dumb", "awful", "terrible", "bad", "angry", "mad", "upset", "annoying", "sad", "sorry" };
        private static readonly string[] CalmingWords  = { "calm", "relax", "breathe", "sleep", "rest", "peace" };
        private static readonly string[] HighArousalMarkers = { "!", "!!!", "now", "hurry", "quick", "urgent" };
        private static readonly string[] LowArousalWords = { "tired", "sleepy", "bored", "boring", "exhausted" };

        // quick-and-dirty text scan to raw pulses
        public Appraisal Appraise(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Appraisal(0f, 0f, 0);

            var lower = text.ToLowerInvariant();

            int posHits = CountHits(lower, PositiveWords);
            int negHits = CountHits(lower, NegativeWords);

            int net = posHits - negHits;
            float valPulse = net == 0 ? 0f : Math.Clamp(net * 0.25f, -1f, 1f);

            float arousalPulse = 0.25f; // low simmer baseline
            if (HighArousalMarkers.Any(m => lower.Contains(m)))
                arousalPulse += 0.25f;
            if (LowArousalWords.Any(w => lower.Contains(w)))
                arousalPulse -= 0.20f;
            if (CalmingWords.Any(w => lower.Contains(w)))
                arousalPulse -= 0.15f;
            if (HasShouting(text))
                arousalPulse += 0.25f;

            arousalPulse = Math.Clamp(arousalPulse, 0f, 1f);
            int pleasantness = (int)Math.Round(valPulse * 10f);

            return new Appraisal(valPulse, arousalPulse, pleasantness);
        }

        public Appraisal Appraise(string text, RelationshipSnapshot relationship, IReadOnlyList<MemoryEntry>? candidateMemories = null)
        {
            var context = new AppraisalContext(
                text,
                Snapshot(),
                relationship,
                candidateMemories ?? Array.Empty<MemoryEntry>());

            return _appraisalProvider.Evaluate(context);
        }

        public void Apply(Appraisal appraisal)
        {
            Valence = Smooth(Valence, appraisal.ValencePulse, ValenceBlend);
            Arousal = Smooth(Arousal, appraisal.ArousalPulse, ArousalBlend);
        }

        public EmotionSnapshot Snapshot() => new EmotionSnapshot(Valence, Arousal, CurrentLabel);

        // nudge back toward relaxed over real elapsed time
        public void Decay(IClock clock)
        {
            var now = clock.UtcNow;
            if (_lastDecayUtc == DateTimeOffset.MinValue)
            {
                _lastDecayUtc = now;
                return;
            }

            var elapsedSeconds = (now - _lastDecayUtc).TotalSeconds;
            if (elapsedSeconds <= 0) return;

            const double blendPerMinute = 0.02;
            var blend = (float)Math.Min(1.0, (elapsedSeconds / 60.0) * blendPerMinute);

            Valence = Smooth(Valence, NeutralTarget.v, blend);
            Arousal = Smooth(Arousal, NeutralTarget.a, blend);

            _lastDecayUtc = now;
        }

        // how close a memory's vibe is to the current vibe
        public float AffectMatch(float memValence, float memArousal)
        {
            var dist = AffectDistance(Valence, Arousal, memValence, memArousal);
            const float roughMax = 1.5f;
            var norm = Math.Clamp(dist / roughMax, 0f, 1f);
            return 1f - norm;
        }

        // helpers
        public static float AffectDistance(float v1, float a1, float v2, float a2)
        {
            var dv = v1 - v2;
            var da = (a1 - a2) * 0.8f; // arousal counts but softer
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

        // simple bucket labels for now
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

    // slap the current mood into a memory snapshot
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
        public float Valence;
        public float Arousal;
        public int Pleasantness;
        public int RelationshipPoints;
        public float MaterialImportance;
        public float[] Embedding = Array.Empty<float>();

        public EmotionSnapshot EmotionSnapshot => new EmotionSnapshot(
            Valence,
            Arousal,
            EmotionEngine.LabelFrom(Valence, Arousal)
        );
    }
}

