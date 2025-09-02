
// Simple Input-Output AI API Service (Terminal-based)
Console.WriteLine("Oddyseus AI Service - Type 'exit' to quit.");
while (true)
{
	Console.Write("You: ");
	string? input = Console.ReadLine();
	if (string.IsNullOrWhiteSpace(input))
		continue;
	if (input.Trim().ToLower() == "exit")
		break;

	// Placeholder for AI logic, just gonna repeat now.
	string response = $"AI: You said '{input}'";
	Console.WriteLine(response);
}
Console.WriteLine("Goodbye!");
