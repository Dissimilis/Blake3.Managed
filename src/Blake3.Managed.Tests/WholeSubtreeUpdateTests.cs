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
}
