using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Oddyseus.Core;

public sealed class MemoryManager : IDisposable
{
    private static string ResolveModelPath()
    {
        var baseDir = AppContext.BaseDirectory;
        var candidates = new[]
        {
            Path.Combine(baseDir, "Core", "Models", "Qwen3EmbModel.onnx"),
            Path.Combine(baseDir, "Oddyseus", "Core", "Models", "Qwen3EmbModel.onnx")
        };

        foreach (var path in candidates)
        {
            if (File.Exists(path))
                return path;
        }

        throw new FileNotFoundException("Embedding model missing.", candidates.Last());
    }

    private readonly InferenceSession _embedSession;

    public MemoryManager()
    {
        _embedSession = new InferenceSession(ResolveModelPath());
    }

    public float[] Embed(ReadOnlyMemory<int> tokenIds)
    {
        if (tokenIds.IsEmpty)
            throw new ArgumentException("Tokenizer returned no tokens.", nameof(tokenIds));

        var tokens = tokenIds.ToArray();
        var longs = new long[tokens.Length];
        var mask = new long[tokens.Length];
        var positions = new long[tokens.Length];

        for (var i = 0; i < tokens.Length; i++)
        {
            longs[i] = tokens[i];
            mask[i] = 1;
            positions[i] = i;
        }

        var inputIds = new DenseTensor<long>(longs, new[] { 1, tokens.Length });
        var attentionMask = new DenseTensor<long>(mask, new[] { 1, tokens.Length });
        var positionIds = new DenseTensor<long>(positions, new[] { 1, tokens.Length });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor("position_ids", positionIds)
        };

        using var results = _embedSession.Run(inputs, _embedSession.OutputNames);
        var output = results.FirstOrDefault()
                     ?? throw new InvalidOperationException("Embedding model returned no output.");
        
        var float16Tensor = output.AsEnumerable<Float16>().ToArray();
        return float16Tensor.Select(f => (float)f).ToArray();
    }

    public void Dispose() => _embedSession.Dispose();

    public readonly record struct MemorySignals(
        float SemanticSimilarity,
        float Valence,
        float Arousal,
        int Pleasantness,
        float RelationshipPoints,
        float MaterialImportance,
        long MonotonicStamp,
        DateTimeOffset Timestamp);

    public float ComputeWeightedScore(in MemorySignals signals, DateTimeOffset now, TimeSpan halfLife, long nowMonotonicTicks)
    {
        const float relationshipWeight = 0.35f;
        const float pleasantnessWeight = 0.30f;
        const float arousalWeight = 0.20f;
        const float importanceWeight = 0.15f;
        const float baselineScore = 0.05f;  // lower floor so semantics drive ranking

        var normalizedRelationship = Math.Clamp(signals.RelationshipPoints / 100f, -1f, 1f);
        var normalizedPleasantness = Math.Clamp(signals.Pleasantness / 10f, -1f, 1f);
        var normalizedArousal = Math.Clamp(signals.Arousal, -1f, 1f);
        var normalizedImportance = Math.Clamp(signals.MaterialImportance, 0f, 1f);

        var affectScore = baselineScore +
            (relationshipWeight * normalizedRelationship) +
            (pleasantnessWeight * MathF.Abs(normalizedPleasantness)) +
            (arousalWeight * MathF.Abs(normalizedArousal)) +
            (importanceWeight * normalizedImportance);

        double elapsedSeconds;
        if (signals.MonotonicStamp > 0 && nowMonotonicTicks > signals.MonotonicStamp)
        {
            var tickDelta = nowMonotonicTicks - signals.MonotonicStamp;
            elapsedSeconds = tickDelta / (double)Stopwatch.Frequency;
        }
        else
        {
            elapsedSeconds = (now - signals.Timestamp).TotalSeconds;
        }

        var decay = MathF.Exp(-(float)(elapsedSeconds / halfLife.TotalSeconds));
        return signals.SemanticSimilarity * affectScore * decay;
    }

    // grab candidates, fuse semantic + affect + decay, kick back best matches
    public IReadOnlyList<(MemoryEntry Memory, float Score)> RetrieveTopMemories(
        ReadOnlySpan<float> queryEmbedding,
        IEnumerable<MemoryEntry> candidates,
        IEmotionEngine emotionEngine,
        int take,
        TimeSpan halfLife,
        DateTimeOffset now,
        long nowMonotonicTicks,
        float minScore = 0f,
        float semanticFloor = 0.15f,
        DateTimeOffset? targetTime = null,
        TimeSpan? timeWindow = null)
    {
        var ranked = new List<(MemoryEntry, float)>();
        var windowSeconds = timeWindow?.TotalSeconds ?? 0;

        foreach (var memory in candidates)
        {
            if (memory.Embedding.Length == 0) continue;

            var semanticSim = CosineSimilarity(queryEmbedding, memory.Embedding);
            if (semanticSim < semanticFloor) continue;

            var signals = new MemorySignals(
                semanticSim,
                memory.Valence,
                memory.Arousal,
                memory.Pleasantness,
                memory.RelationshipPoints,
                memory.MaterialImportance,
                memory.MonotonicStamp,
                memory.TimeUtc);

            var baseScore = ComputeWeightedScore(signals, now, halfLife, nowMonotonicTicks);
            var affectAlignment = emotionEngine.AffectMatch(memory.Valence, memory.Arousal);
            var totalScore = baseScore * affectAlignment;

            if (targetTime.HasValue && windowSeconds > 0)
            {
                var delta = Math.Abs((targetTime.Value - memory.TimeUtc).TotalSeconds);
                var timeProximity = MathF.Exp(-(float)(delta / windowSeconds));
                totalScore *= (0.5f + timeProximity); // modest boost for closer timestamps
            }

            Console.WriteLine($"Memory '{memory.UserText}': sem={semanticSim:F3}, base={baseScore:F3}, affect={affectAlignment:F3}, total={totalScore:F3}");

            if (totalScore < minScore) continue;

            ranked.Add((memory, totalScore));
        }

        return ranked
            .OrderByDescending(tuple => tuple.Item2)
            .Take(take)
            .ToList();
    }

    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var length = Math.Min(a.Length, b.Length);
        double dot = 0;
        double magA = 0;
        double magB = 0;

        for (int i = 0; i < length; i++)
        {
            var av = a[i];
            var bv = b[i];
            dot += av * bv;
            magA += av * av;
            magB += bv * bv;
        }

        if (magA == 0 || magB == 0) return 0f;
        return (float)(dot / (Math.Sqrt(magA) * Math.Sqrt(magB)));
    }
}