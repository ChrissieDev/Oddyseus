using Oddyseus.Core;
using Oddyseus.Oddyseus.Core;

var tokenizer = new BasicTokenizer();
using var memory = new MemoryManager();
var emotion = new EmotionEngine();
var relationships = new RelationshipModeler();

var appraisalEndpoint = "https://api.groq.com/openai/v1";
var responseEndpoint = "https://api.groq.com/openai/v1";
var apiKey = Environment.GetEnvironmentVariable("GROQ_API_KEY") ?? "gsk-test";

using var appraisalClient = new LlmClient("https://api.groq.com/openai/v1/chat/completions", apiKey, "llama-3.1-8b-instant");
using var responseClient = new LlmClient("https://api.groq.com/openai/v1/chat/completions", apiKey, "llama-3.1-8b-instant");

var orchestrator = new Orchestrator(memory, emotion, relationships, tokenizer, appraisalClient, responseClient);

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
