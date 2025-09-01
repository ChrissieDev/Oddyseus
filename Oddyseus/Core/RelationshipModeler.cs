using System;
using System.Collections.Generic;
using Oddyseus.Types;

namespace Oddyseus.Core
{
    public class RelationshipModeler : IRelationshipModeler
    {
        private readonly Dictionary<string, UserRelationship> _relationships = new();

        public void AddInteraction(string userName)
        {
            if (!_relationships.TryGetValue(userName, out var rel))
            {
                rel = new UserRelationship(userName);
                _relationships[userName] = rel;
            }
            rel.InteractionCount++;
        }

        // Interface requirement
        public void AdjustPoints(string user, int delta)
        {
            if (!_relationships.TryGetValue(user, out var rel))
            {
                rel = new UserRelationship(user);
                _relationships[user] = rel;
            }

            // Prefer a method if it exists, or else clamp manually.
            rel.UpdateRelationshipPoints(delta);
        }

        // Interface requirement
        public RelationshipData GetRelationshipData(string user)
        {
            if (!_relationships.TryGetValue(user, out var rel))
            {
                rel = new UserRelationship(user);
                _relationships[user] = rel;
            }

            return new RelationshipData
            {
                UserName = rel.UserName,
                InteractionCount = rel.InteractionCount,
                RelationshipPoints = rel.RelationshipPoints,
            };
        }


        public class UserRelationship
        {
            public string UserName { get; private set; }
            public int InteractionCount { get; set; }
            public int RelationshipPoints { get; private set; } = 0; // -100 to 100

            public UserRelationship(string userName)
            {
                UserName = userName;
                InteractionCount = 0;
            }

            public double CalculateAffinity()
            {
                return InteractionCount > 0 ? Math.Log(InteractionCount) : 0.0;
            }

            public void UpdateRelationshipPoints(int delta)
            {
                RelationshipPoints = Math.Max(-100, Math.Min(100, RelationshipPoints + delta));
            }

            public string GetRelationshipPartition(int points)
            {
                // points: -100 to 100
                if (points <= -84) return "ExtremeDislike";
                if (points <= -67) return "StrongDislike";
                if (points <= -50) return "Dislike";
                if (points <= -34) return "MildDislike";
                if (points <= -17) return "Uneasy";
                if (points < 0) return "Distant";
                if (points == 0) return "Neutral";
                if (points <= 17) return "MildLike";
                if (points <= 34) return "Like";
                if (points <= 50) return "StrongLike";
                if (points <= 67) return "Admiration";
                if (points <= 84) return "Affection";
                return "Love";
            }
        }
    }
}