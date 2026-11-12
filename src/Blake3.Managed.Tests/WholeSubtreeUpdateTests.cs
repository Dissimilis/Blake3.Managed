namespace Blake3.Managed.Tests;

/// <summary>
/// Updates whose input is exactly an aligned power-of-two subtree, so the subtree's right half is
/// deferred rather than merged, followed by more input that has to flush it at the right size.
/// </summary>
public class WholeSubtreeUpdateTests
{
    public static IEnumerable<object[]> Splits()
    {
        int[][] cases =
        {
            new[] { 16 * 1024 },
            new[] { 32 * 1024 },
            new[] { 64 * 1024 },
            new[] { 128 * 1024 },
            new[] { 16 * 1024, 1 },
            new[] { 16 * 1024, 16 * 1024 },
            new[] { 16 * 1024, 16 * 1024, 32 * 1024 },
            new[] { 32 * 1024, 5 },
            new[] { 64 * 1024, 64 * 1024, 7 },
            new[] { 8 * 1024, 8 * 1024, 16 * 1024 },
            new[] { 1024, 16 * 1024 },
            new[] { 16 * 1024, 1024, 15 * 1024 },
            new[] { 64 * 1024, 1024 },
            new[] { 128 * 1024, 64 * 1024, 32 * 1024, 16 * 1024 },
            new[] { 40 * 1024 },
            new[] { 48 * 1024 },
            new[] { 72 * 1024, 3 },
            new[] { 32 * 1024, 32 * 1024 },
            new[] { 32 * 1024, 40 * 1024, 1024 },
            new[] { 256 * 1024 },
            new[] { 264 * 1024, 8 * 1024 },
            new[] { 1024 * 1024, 64 * 1024 },
            new[] { 16 * 1024, 48 * 1024, 64 * 1024, 1 },
        };
        foreach (var c in cases) yield return new object[] { c };
    }

    [Theory]
    [MemberData(nameof(Splits))]
    public void MatchesOneShotForDefaultKeyedAndXof(int[] parts)
    {
        int total = parts.Sum();
        byte[] input = HasherTests.MakeTestInput(total);
        byte[] key = HasherTests.MakeTestInput(32);

        using var hasher = Hasher.New();
        using var keyed = Hasher.NewKeyed(key);
        using var keyedWhole = Hasher.NewKeyed(key);
        using var join = Hasher.New();
        int pos = 0;
        foreach (int n in parts)
        {
            hasher.Update(input.AsSpan(pos, n));
            keyed.Update(input.AsSpan(pos, n));
            join.UpdateWithJoin(input.AsSpan(pos, n));
            pos += n;
        }
        keyedWhole.Update(input);

        Assert.Equal(Hasher.Hash(input), hasher.Finalize());
        Assert.Equal(Hasher.Hash(input), join.Finalize());
        Assert.Equal(keyedWhole.Finalize(), keyed.Finalize());

        byte[] xofExpected = new byte[300];
        byte[] xofActual = new byte[300];
        using var oneShotXof = Hasher.New();
        oneShotXof.Update(input);
        oneShotXof.Finalize(xofExpected);
        hasher.Finalize(xofActual);
        Assert.Equal(xofExpected, xofActual);
    }

    /// <summary>
    /// Random sequences of <c>UpdateWithJoin</c> calls, most of them whole multiples of 8 KiB
    /// (so the last unit is deferred), mixed with odd lengths and plain <c>Update</c> calls.
    /// </summary>
    [Fact]
    public void RandomJoinSequencesMatchOneShot()
    {
        var random = new Random(20261112);
        byte[] key = HasherTests.MakeTestInput(32);
        byte[] xofExpected = new byte[200];
        byte[] xofActual = new byte[200];

        for (int round = 0; round < 300; round++)
        {
            int count = random.Next(1, 5);
            int[] parts = new int[count];
            for (int i = 0; i < count; i++)
            {
                parts[i] = random.Next(6) switch
                {
                    0 => random.Next(1, 70_000),
                    1 => random.Next(1, 300) * 1024,
                    _ => random.Next(1, 48) * 8 * 1024,
                };
            }

            byte[] input = HasherTests.MakeTestInput(parts.Sum());
            using var join = Hasher.New();
            using var keyed = Hasher.NewKeyed(key);
            int pos = 0;
            for (int i = 0; i < count; i++)
            {
                join.UpdateWithJoin(input.AsSpan(pos, parts[i]));
                if (random.Next(4) == 0) keyed.Update(input.AsSpan(pos, parts[i]));
                else keyed.UpdateWithJoin(input.AsSpan(pos, parts[i]));
                pos += parts[i];
            }

            string shape = string.Join(",", parts);
            Assert.True(Hasher.Hash(input) == join.Finalize(), shape);
            Assert.True(Hasher.HashKeyed(key, input) == keyed.Finalize(), shape);

            using var whole = Hasher.New();
            whole.Update(input);
            whole.Finalize(xofExpected);
            join.Finalize(xofActual);
            Assert.True(xofExpected.AsSpan().SequenceEqual(xofActual), shape);
        }
    }
}
