using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Oddyseus.Oddyseus.Core;
using Oddyseus.Types;

namespace Oddyseus.Core;

public interface ILlmClient
{
    Task<JsonDocument> CompleteJsonAsync(string systemPrompt, string userPrompt, CancellationToken ct);
}

public sealed class LlmClient : ILlmClient, IDisposable
{
    private readonly HttpClient _http;
    private readonly string _model;
    private readonly string _fullUrl;

    public LlmClient(string fullUrl, string apiKey, string model)
    {
        _http = new HttpClient();
        _http.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        _fullUrl = fullUrl;
        _model = model;
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

        Console.WriteLine($"DEBUG: Posting to {_fullUrl}");
        using var response = await _http.PostAsJsonAsync(_fullUrl, body, ct);
        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct);
            Console.WriteLine($"ERROR: {(int)response.StatusCode} - {errorBody}");
            response.EnsureSuccessStatusCode();
        }

        var responseText = await response.Content.ReadAsStringAsync(ct);
        Console.WriteLine($"DEBUG: Raw response: {responseText}");
        
        using var fullJson = JsonDocument.Parse(responseText);
        var contentString = fullJson.RootElement
            .GetProperty("choices")[0]
            .GetProperty("message")
            .GetProperty("content")
            .GetString();
        
        Console.WriteLine($"DEBUG: Extracted content: {contentString}");
        return JsonDocument.Parse(contentString);
    }

    public void Dispose() => _http.Dispose();
}

public interface ITokenizer
{
    int[] Encode(string text);
}

public sealed class BasicTokenizer : ITokenizer
{
    public int[] Encode(string text) =>
        string.IsNullOrEmpty(text) ? Array.Empty<int>() : text.Select(ch => (int)ch).ToArray();
}

public sealed class Orchestrator
{
    private readonly MemoryManager _memory;
    private readonly IEmotionEngine _emotion;
    private readonly RelationshipModeler _relationships;
    private readonly ILlmClient _appraisalClient;
    private readonly ILlmClient _responseClient;
    private readonly ITokenizer _tokenizer;
    private readonly List<MemoryEntry> _memoryBank = new();
    private readonly List<(string Role, string Text)> _dialogue = new();

    public Orchestrator(
        MemoryManager memory,
        IEmotionEngine emotion,
        RelationshipModeler relationships,
        ITokenizer tokenizer,
        ILlmClient appraisalClient,
        ILlmClient responseClient)
    {
        _memory = memory;
        _emotion = emotion;
        _relationships = relationships;
        _tokenizer = tokenizer;
        _appraisalClient = appraisalClient;
        _responseClient = responseClient;
    }

    public async Task<string> RunTurnAsync(string userName, string userText, CancellationToken ct = default)
    {
        var tokens = _tokenizer.Encode(userText);
        var embedding = _memory.Embed(tokens);
        var relationship = _relationships.GetRelationshipData(userName);
        var now = DateTimeOffset.UtcNow;

        var curated = _memory
            .RetrieveTopMemories(embedding, _memoryBank, _emotion, 5, TimeSpan.FromHours(6), now)
            .Select(pair => pair.Memory)
            .ToList();

        var appraisalResult = await RequestAppraisalAsync(userText, relationship, curated, ct);
        _emotion.Apply(appraisalResult.Appraisal);
        _relationships.AdjustPoints(userName, appraisalResult.Appraisal.Pleasantness);

        var response = await RequestResponseAsync(userText, curated, appraisalResult.Appraisal, relationship, ct);

        var entry = new MemoryEntry
        {
            Role = "user",
            UserText = userText,
            AiText = response,
            Pleasantness = appraisalResult.Appraisal.Pleasantness,
            RelationshipPoints = relationship.RelationshipPoints,
            MaterialImportance = appraisalResult.MaterialImportance,
            Embedding = embedding,
            TimeUtc = now.UtcDateTime
        };
        entry.StampEmotion(_emotion);
        _memoryBank.Add(entry);

        _dialogue.Add(("user", userText));
        _dialogue.Add(("assistant", response));
        if (_dialogue.Count > 8)
            _dialogue.RemoveRange(0, _dialogue.Count - 8);

        return response;
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
            "Return ONLY valid JSON (no markdown, no text) with keys: valence (-1..1), arousal (0..1), pleasantness (-10..10), material_importance (0..1).",
            JsonSerializer.Serialize(payload),
            ct);

        try
        {
            var root = json.RootElement;
            var valence = root.GetProperty("valence").GetSingle();
            var arousal = root.GetProperty("arousal").GetSingle();
            var pleasantness = root.GetProperty("pleasantness").GetInt32();
            var material = root.TryGetProperty("material_importance", out var mi) ? mi.GetSingle() : 0f;
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
        CancellationToken ct)
    {
        var recalledTexts = new HashSet<string>(
            curated.SelectMany(m => new[] { m.UserText, m.AiText })
                   .Where(s => !string.IsNullOrWhiteSpace(s)));

        var recentTurns = _dialogue
            .Skip(Math.Max(0, _dialogue.Count - 8))
            .Where(turn => !recalledTexts.Contains(turn.Text))
            .ToList();

        var prompt = new
        {
            user_text = userText,
            emotion = new { appraisal.ValencePulse, appraisal.ArousalPulse, appraisal.Pleasantness },
            relationship = relationship,
            curated_memories = curated.Select(m => new { m.Id, m.UserText, m.AiText }),
            recent_dialogue = recentTurns.Select(t => new { t.Role, t.Text })
        };

        using var json = await _responseClient.CompleteJsonAsync(
            "Respond as the companion using provided memories; return {\"reply\":\"...\"}.",
            JsonSerializer.Serialize(prompt),
            ct);

        return json.RootElement.GetProperty("reply").GetString() ?? "(no reply)";
    }
}