using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Diagnostics;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Oddyseus.Oddyseus.Core;
using Oddyseus.Types;
using Tokenizers.DotNet;

namespace Oddyseus.Core;

public interface ILlmClient
{
    Task<JsonDocument> CompleteJsonAsync(string systemPrompt, string userPrompt, CancellationToken ct);
    Task<string> CompleteTextAsync(string systemPrompt, string userPrompt, CancellationToken ct);
}

public sealed class LlmClient : ILlmClient, IDisposable
{
    private readonly HttpClient _http;
    private readonly string _model;
    private readonly string _fullUrl;
    private const int MaxAttempts = 3;

    public LlmClient(string fullUrl, string apiKey, string model)
    {
        _http = new HttpClient();
        _http.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        _fullUrl = fullUrl;
        _model = model;
    }

    private static TimeSpan ComputeDelay(int attempt, TimeSpan? retryAfter)
    {
        if (retryAfter.HasValue && retryAfter.Value > TimeSpan.Zero)
            return retryAfter.Value;
        var backoff = Math.Pow(2, attempt - 1);
        return TimeSpan.FromSeconds(1.0 * backoff);
    }

    private async Task<string> PostWithRetryAsync(object body, CancellationToken ct)
    {
        for (int attempt = 1; attempt <= MaxAttempts; attempt++)
        {
            using var response = await _http.PostAsJsonAsync(_fullUrl, body, ct);

            if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests && attempt < MaxAttempts)
            {
                var retryAfter = response.Headers.RetryAfter?.Delta;
                var wait = ComputeDelay(attempt, retryAfter);
                Console.WriteLine($"Rate limited (429). Retrying in {wait.TotalSeconds:F1}s...");
                await Task.Delay(wait, ct);
                continue;
            }

            if (!response.IsSuccessStatusCode)
            {
                var errorBody = await response.Content.ReadAsStringAsync(ct);
                Console.WriteLine($"ERROR: {(int)response.StatusCode} - {errorBody}");
                response.EnsureSuccessStatusCode();
            }

            return await response.Content.ReadAsStringAsync(ct);
        }

        throw new InvalidOperationException("Exhausted retries calling LLM endpoint.");
    }

    private static string NormalizeJsonString(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
            return "{}";

        // Replace common smart quotes with standard quotes to avoid JSON parse errors from model output
        return raw
            .Replace('\u201c', '"')
            .Replace('\u201d', '"')
            .Replace('\u2018', '\'')
            .Replace('\u2019', '\'');
    }

    public async Task<JsonDocument> CompleteJsonAsync(string systemPrompt, string userPrompt, CancellationToken ct)
    {
        var body = new
        {
            model = _model,
            messages = new[]
            {
                new { role = "system", content = systemPrompt },
                new { role = "user", content = userPrompt }
            }
        };

        var responseText = await PostWithRetryAsync(body, ct);
        using var fullJson = JsonDocument.Parse(responseText);
        var contentString = fullJson.RootElement
            .GetProperty("choices")[0]
            .GetProperty("message")
            .GetProperty("content")
            .GetString();
        
        try
        {
            var normalized = NormalizeJsonString(contentString);
            return JsonDocument.Parse(normalized);
        }
        catch (JsonException ex)
        {
            Console.WriteLine($"LLM returned non-JSON: {contentString}");
            throw new InvalidOperationException($"LLM did not return JSON: {contentString}", ex);
        }
    }

    public async Task<string> CompleteTextAsync(string systemPrompt, string userPrompt, CancellationToken ct)
    {
        var body = new
        {
            model = _model,
            messages = new[]
            {
                new { role = "system", content = systemPrompt },
                new { role = "user", content = userPrompt }
            }
        };

        var responseText = await PostWithRetryAsync(body, ct);
        using var fullJson = JsonDocument.Parse(responseText);
        var contentString = fullJson.RootElement
            .GetProperty("choices")[0]
            .GetProperty("message")
            .GetProperty("content")
            .GetString();

        return contentString ?? string.Empty;
    }

    public void Dispose() => _http.Dispose();
}

public interface ITokenizer
{
    int[] Encode(string text);
}

public sealed class QwenTokenizer : ITokenizer
{
    public static string ResolvePath()
    {
        var baseDir = AppContext.BaseDirectory;
        var candidates = new[]
        {
            Path.Combine(baseDir, "Core", "Models", "tokenizer.json"),
            Path.Combine(baseDir, "Oddyseus", "Core", "Models", "tokenizer.json")
        };

        foreach (var path in candidates)
        {
            if (File.Exists(path))
                return path;
        }

        throw new FileNotFoundException("Tokenizer file missing.", candidates.Last());
    }

    private readonly Tokenizer _tokenizer;

    public QwenTokenizer(string tokenizerPath)
    {
        _tokenizer = new Tokenizer(vocabPath: tokenizerPath);
    }

    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
            return Array.Empty<int>();

        var ids = _tokenizer.Encode(text);
        return ids.Select(id => (int)id).ToArray();
    }
}

public sealed class Orchestrator
{
    private readonly MemoryManager _memory;
    private readonly IEmotionEngine _emotion;
    private readonly RelationshipModeler _relationships;
    private readonly ILlmClient _appraisalClient;
    private readonly ILlmClient _responseClient;
    private readonly ITokenizer _tokenizer;
    private readonly IClock _clock;
    private readonly DateTimeOffset _sessionStart;
    private readonly List<MemoryEntry> _memoryBank = new();
    private readonly List<(string Role, string Text)> _dialogue = new();

    private static string FormatAge(TimeSpan delta)
    {
        if (delta < TimeSpan.Zero)
            delta = TimeSpan.Zero;

        if (delta.TotalSeconds < 90)
            return $"{delta.TotalSeconds:F0}s ago";
        if (delta.TotalMinutes < 90)
            return $"{delta.TotalMinutes:F0}m ago";
        if (delta.TotalHours < 48)
            return $"{delta.TotalHours:F0}h ago";
        if (delta.TotalDays < 14)
            return $"{delta.TotalDays:F0}d ago";
        return $"{delta.TotalDays / 7:F0}w ago";
    }

    public Orchestrator(
        MemoryManager memory,
        IEmotionEngine emotion,
        RelationshipModeler relationships,
        ITokenizer tokenizer,
        ILlmClient appraisalClient,
        ILlmClient responseClient,
        IClock clock)
    {
        _memory = memory;
        _emotion = emotion;
        _relationships = relationships;
        _tokenizer = tokenizer;
        _appraisalClient = appraisalClient;
        _responseClient = responseClient;
        _clock = clock;
        _sessionStart = clock.UtcNow;
    }

    public async Task<string> RunTurnAsync(string userName, string userText, CancellationToken ct = default)
    {
        // embed the live user query for retrieval
        var embedSw = Stopwatch.StartNew();
        var queryTokens = _tokenizer.Encode(userText);
        var queryEmbedding = _memory.Embed(queryTokens);
        embedSw.Stop();
        Console.WriteLine($"DEBUG: Embed latency {embedSw.ElapsedMilliseconds} ms for {queryTokens.Length} tokens");
        var relationship = _relationships.GetRelationshipData(userName);
        var now = _clock.UtcNow;
        var nowMonotonic = _clock.MonotonicTicks;

        var (targetTime, timeWindow) = DetectTimeHint(userText, now);

        const float minScore = 0.0f; // allow fallback memories even when fresh bank is sparse
        const float semanticFloor = 0.15f;
        var ranked = _memory
            .RetrieveTopMemories(queryEmbedding, _memoryBank, _emotion, 5, TimeSpan.FromHours(6), now, nowMonotonic, minScore, semanticFloor, targetTime, timeWindow);

        var curated = ranked
            .Select(pair => pair.Memory)
            .ToList();

        // If nothing met the score threshold, fall back to the most recent memory to keep temporal grounding
        if (curated.Count == 0 && _memoryBank.Count > 0)
        {
            var fallback = _memoryBank.OrderByDescending(m => m.TimeUtc).First();
            curated.Add(fallback);
            Console.WriteLine("DEBUG: No memories passed minScore; using most recent memory as fallback.");
        }

        Console.WriteLine($"DEBUG: Ranked {ranked.Count} memories, kept {curated.Count} (>= {minScore:F2}) from bank of {_memoryBank.Count}");
        foreach (var (memory, score) in ranked)
            Console.WriteLine($"  - {memory.UserText} | Score: {score:F3}");

        var appraisalResult = await RequestAppraisalAsync(userText, relationship, curated, ct);
        Console.WriteLine($"Appraisal: valence={appraisalResult.Appraisal.ValencePulse}, arousal={appraisalResult.Appraisal.ArousalPulse}, pleasantness={appraisalResult.Appraisal.Pleasantness}");

        _emotion.Apply(appraisalResult.Appraisal);
        _relationships.AdjustPoints(userName, appraisalResult.Appraisal.Pleasantness);

        var response = await RequestResponseAsync(userText, curated, appraisalResult.Appraisal, relationship, now, ct);

        // embed the full exchange with timestamp for future recall
        var memoryText = $"[{now:O}] User: {userText}\nAssistant: {response}";
        var memoryTokens = _tokenizer.Encode(memoryText);
        embedSw.Restart();
        var memoryEmbedding = _memory.Embed(memoryTokens);
        embedSw.Stop();
        Console.WriteLine($"DEBUG: Persisted embed latency {embedSw.ElapsedMilliseconds} ms for {memoryTokens.Length} tokens"); // just debug

        var entry = new MemoryEntry
        {
            Role = "user",
            UserText = userText,
            AiText = response,
            Pleasantness = appraisalResult.Appraisal.Pleasantness,
            RelationshipPoints = relationship.RelationshipPoints,
            MaterialImportance = appraisalResult.MaterialImportance,
            Embedding = memoryEmbedding,
            TimeUtc = now,
            MonotonicStamp = nowMonotonic
        };
        entry.StampEmotion(_emotion);
        _memoryBank.Add(entry);


        // Add to dialogue buffer
        _dialogue.Add(("user", userText));
        _dialogue.Add(("assistant", response));
        // Summarize if buffer is too long
        const int maxDialogueTurns = 16;
        const int summaryTurns = 8;
        if (_dialogue.Count > maxDialogueTurns)
        {
            // Take the oldest N turns and summarize
            var toSummarize = _dialogue.Take(summaryTurns).ToList();
            var summaryText = string.Join("\n", toSummarize.Select(t => $"[{t.Role}] {t.Text}"));
            var summaryPrompt = $"Summarize the following conversation turns into a concise, factual, and emotionally aware summary. Focus on key facts, decisions, and emotional tone.\n\n{summaryText}";
            var summary = await _responseClient.CompleteTextAsync(
                "You are a summarizer for an AI memory system. Return a short, factual, emotionally aware summary of the provided turns.",
                summaryPrompt,
                ct);
            // Replace the oldest N turns with the summary
            _dialogue.RemoveRange(0, summaryTurns);
            _dialogue.Insert(0, ("summary", summary.Trim()));
        }
        if (_dialogue.Count > maxDialogueTurns)
            _dialogue.RemoveRange(0, _dialogue.Count - maxDialogueTurns);

        return response;
    }

    private (DateTimeOffset? targetTime, TimeSpan? window) DetectTimeHint(string userText, DateTimeOffset now)
    {
        var lower = userText.ToLowerInvariant();

        if (lower.Contains("first interaction") || lower.Contains("first time we") || lower.Contains("very beginning") ||
            lower.Contains("very start") || lower.Contains("initially") || lower.Contains("from the start") || lower.Contains("from the beginning"))
        {
            if (_memoryBank.Count > 0)
            {
                var earliest = _memoryBank.Min(m => m.TimeUtc);
                return (earliest, TimeSpan.FromHours(12));
            }
            return (null, null);
        }

        if (lower.Contains("hour ago"))
            return (now.AddHours(-1), TimeSpan.FromHours(1.5));

        if (lower.Contains("yesterday"))
            return (now.AddDays(-1), TimeSpan.FromHours(8));

        if (lower.Contains("last week"))
            return (now.AddDays(-7), TimeSpan.FromDays(3));

        if (lower.Contains("today"))
            return (now, TimeSpan.FromHours(6));

        return (null, null);
    }

    private async Task<(Appraisal Appraisal, float MaterialImportance)> RequestAppraisalAsync(
        string userText,
        RelationshipData relationship,
        IReadOnlyList<MemoryEntry> curated,
        CancellationToken ct)
    {
        var payload = new
        {
            input = userText,
            relationship_points = relationship.RelationshipPoints,
            mood = new { valence = _emotion.Valence, arousal = _emotion.Arousal },
            memories = curated.Select(m => new { m.Id, m.UserText, m.AiText, m.Pleasantness, m.MaterialImportance })
        };

        using var json = await _appraisalClient.CompleteJsonAsync(
            "You MUST respond with ONLY a JSON object. Do not include any text before or after. Return exactly: {\"valence\":NUMBER,-1 to 1,\"arousal\":NUMBER,0 to 1,\"pleasantness\":INTEGER,-10 to 10,\"material_importance\":NUMBER,0 to 1}. If the input is neutral or playful, keep pleasantness near 0. Only use large negative pleasantness for clearly hostile or harmful user intent.",
            JsonSerializer.Serialize(payload),
            ct);

        try
        {
            var root = json.RootElement;
            var valence = root.TryGetProperty("valence", out var v) ? v.GetSingle() : 0f;
            var arousal = root.TryGetProperty("arousal", out var a) ? a.GetSingle() : 0.25f;
            var pleasantness = root.TryGetProperty("pleasantness", out var p) 
                ? (int)Math.Round(p.GetSingle()) 
                : 0;
            var material = root.TryGetProperty("material_importance", out var mi) ? mi.GetSingle() : 0f;

            valence = Math.Clamp(valence, -1f, 1f);
            arousal = Math.Clamp(arousal, 0f, 1f);
            pleasantness = Math.Clamp(pleasantness, -5, 5);
            material = Math.Clamp(material, 0f, 1f);
            return (new Appraisal(valence, arousal, pleasantness), material);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Appraisal parse error: {ex.Message}");
            return (new Appraisal(0f, 0.25f, 0), 0f);
        }
    }

    private async Task<string> RequestResponseAsync(
        string userText,
        IReadOnlyList<MemoryEntry> curated,
        Appraisal appraisal,
        RelationshipData relationship,
        DateTimeOffset now,
        CancellationToken ct)
    {
        var memoryContext = string.Join("\n", curated.Select(m => 
            $"[Memory @ {m.TimeUtc:O} ({FormatAge(now - m.TimeUtc)})] User: {m.UserText} | Me: {m.AiText}"));

        var recentTurns = _dialogue
            .TakeLast(8)
            .Select(t => $"[{t.Role}] {t.Text}");
        var recentContext = string.Join("\n", recentTurns);

        var systemPrompt = "You are a thoughtful AI companion. The emotional state values provided are YOUR current state, not the user's. Use them to modulate your tone (not to claim how the user feels). Respond naturally, realistically, and emotionally, taking relationship into account. When asked about time based inquiries, rely on the provided timestamps, session start, and relative ages instead of guessing.";
        
        var userPrompt = $@"Current time (UTC): {now:O}
    Session start (UTC): {_sessionStart:O} ({FormatAge(now - _sessionStart)} ago)
    Emotional State: valence={appraisal.ValencePulse:F2}, arousal={appraisal.ArousalPulse:F2}, pleasantness={appraisal.Pleasantness}
Relationship: {relationship.RelationshipPoints} points
Background: {(memoryContext.Length > 0 ? memoryContext : "(no prior memories)")}
Recent turns: {(recentContext.Length > 0 ? recentContext : "(none)")}

User says: {userText}

Respond naturally:";

        var content = await _responseClient.CompleteTextAsync(systemPrompt, userPrompt, ct);
        return string.IsNullOrWhiteSpace(content) ? "I'm thinking..." : content;
    }
}