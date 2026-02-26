using System;

namespace Blake3.Managed.Tests;

public class StressTests
{
    private const int DataSize = 8 * 1024 * 1024; // 8 MB
    private const int Iterations = 100;

    private static byte[] MakeLargeData()
    {
        var data = new byte[DataSize];
        Random.Shared.NextBytes(data);
        return data;
    }

    [Fact]
    public void LargeData_RepeatedUpdateWithJoin_NoLeakAndConsistent()
    {
        var data = MakeLargeData();

        string expected;
        using (var h = Hasher.New())
        {
            h.UpdateWithJoin(data);
            expected = h.Finalize().ToString();
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var baselineMemory = GC.GetTotalMemory(forceFullCollection: true);

        for (var i = 0; i < Iterations; i++)
        {
            using var hasher = Hasher.New();
            hasher.UpdateWithJoin(data);
            Assert.Equal(expected, hasher.Finalize().ToString());
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var afterMemory = GC.GetTotalMemory(forceFullCollection: true);

        var growth = afterMemory - baselineMemory;
        Assert.True(growth < 10 * 1024 * 1024,
            $"Possible memory leak: grew {growth / (1024 * 1024)} MB after {Iterations} iterations of UpdateWithJoin");
    }

    [Fact]
    public void LargeData_AllModesProduceSameHash()
    {
        var data = MakeLargeData();

        var oneShot = Hasher.Hash(data).ToString();

        using var hUpdate = Hasher.New();
        hUpdate.Update(data);
        var update = hUpdate.Finalize().ToString();

        using var hJoin = Hasher.New();
        hJoin.UpdateWithJoin(data);
        var join = hJoin.Finalize().ToString();

        Assert.Equal(oneShot, update);
        Assert.Equal(oneShot, join);
    }
}
