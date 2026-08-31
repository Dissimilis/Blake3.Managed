using System;

namespace Blake3.Managed.Tests;

/// <summary>
/// The parallelism cap is a scheduling hint only: every degree must produce the digest the
/// incremental path produces, at every size that reaches the thread-pool tree. Degree 1 is
/// the interesting one, since it routes to the serial tree rather than to a one-thread pool.
/// </summary>
public class MaxDegreeOfParallelismTests
{
    // 268435456 is the regression case: degree * 4 * 2 overflows int, and before the sizing
    // clamp it wrapped the unit-growth bound negative and faulted in the root fold.
    public static TheoryData<int> Degrees => new() { -1, 1, 2, 3, 8, 64, 268_435_456, int.MaxValue };

    private static byte[] Fill(int length)
    {
        var data = new byte[length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (byte)(i * 7 + 1);
        }

        return data;
    }

    /// <summary>Reference digest, taken through the incremental path, which the cap never touches.</summary>
    private static Hash Reference(byte[] data)
    {
        using var hasher = Hasher.New();
        hasher.Update(data);
        return hasher.Finalize();
    }

    [Theory]
    [MemberData(nameof(Degrees))]
    public void AnyDegreeMatchesTheIncrementalDigest(int degree)
    {
        int[] sizes =
        {
            72 * 1024 + 1,      // just past the serial tree's range
            128 * 1024,
            1024 * 1024 + 3,    // ragged tail unit
            4 * 1024 * 1024,
        };

        int original = Hasher.MaxDegreeOfParallelism;
        try
        {
            foreach (int size in sizes)
            {
                var data = Fill(size);
                var expected = Reference(data);

                Hasher.MaxDegreeOfParallelism = degree;
                Assert.Equal(expected, Hasher.Hash(data));

                // The extended-output overload dispatches separately. Its oracle also comes from
                // the incremental path: comparing against this same all-at-once code at degree -1
                // would pass even if the folding or XOF tail were wrong at every degree.
                var wide = new byte[131];
                var expectedWide = new byte[131];
                using (var hasher = Hasher.New())
                {
                    hasher.Update(data);
                    hasher.Finalize(expectedWide);
                }

                Hasher.Hash(data, wide);
                Assert.Equal(expectedWide, wide);
            }
        }
        finally
        {
            Hasher.MaxDegreeOfParallelism = original;
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-2)]
    [InlineData(int.MinValue)]
    public void InvalidDegreeThrows(int degree)
    {
        int original = Hasher.MaxDegreeOfParallelism;
        Assert.Throws<ArgumentOutOfRangeException>(() => Hasher.MaxDegreeOfParallelism = degree);
        Assert.Equal(original, Hasher.MaxDegreeOfParallelism);
    }
}
