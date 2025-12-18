using Oddyseus.Core;
using Oddyseus.Oddyseus.Core;

var tokenizer = new QwenTokenizer(QwenTokenizer.ResolvePath());
using var memory = new MemoryManager();
var emotion = new EmotionEngine();
var relationships = new RelationshipModeler();
var clock = new SystemClock();

var appraisalEndpoint = "https://api.groq.com/openai/v1";
var responseEndpoint = "https://api.groq.com/openai/v1";
var apiKey = Environment.GetEnvironmentVariable("GROQ_API_KEY") ?? "gsk-test";

using var appraisalClient = new LlmClient("https://api.groq.com/openai/v1/chat/completions", apiKey, "llama-3.1-8b-instant");
using var responseClient = new LlmClient("https://api.groq.com/openai/v1/chat/completions", apiKey, "llama-3.1-8b-instant");

var orchestrator = new Orchestrator(memory, emotion, relationships, tokenizer, appraisalClient, responseClient, clock);

// quick embedding sanity check
string s1 = "hello world";
string s2 = "the cat sat on the mat";
string s3 = "financial statements for Q4";
var e1 = memory.Embed(tokenizer.Encode(s1));
var e2 = memory.Embed(tokenizer.Encode(s2));
var e3 = memory.Embed(tokenizer.Encode(s3));
Console.WriteLine("Embedding cosine checks:");
Console.WriteLine($"s1 vs s2: {MemoryManager.CosineSimilarity(e1, e2):F3}");
Console.WriteLine($"s1 vs s3: {MemoryManager.CosineSimilarity(e1, e3):F3}");
Console.WriteLine($"s2 vs s3: {MemoryManager.CosineSimilarity(e2, e3):F3}");

Console.WriteLine("Oddyseus AI Service - Type 'exit' to quit.");
while (true)
{
    Console.Write("You: ");
    var input = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(input))
        continue;
    if (input.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
        break;

    var reply = await orchestrator.RunTurnAsync("default-user", input);
    Console.WriteLine($"AI: {reply}");
}

Console.WriteLine("Goodbye!");
