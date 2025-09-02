using System;

namespace Oddyseus.Core
{
    public interface IClock
    {
        DateTimeOffset UtcNow { get; }
        long MonotonicTicks { get; } // for durations
    }

    public sealed class SystemClock : IClock
    {
        public DateTimeOffset UtcNow => DateTimeOffset.UtcNow;
        public long MonotonicTicks => System.Diagnostics.Stopwatch.GetTimestamp();
    }
}