using System.Runtime.InteropServices;

namespace Blake3.Managed.Tests;

public class DisposeClearsStateTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(64)]
    [InlineData(1500)]
    [InlineData(100 * 1024 + 7)]
    public void DisposeLeavesNoKeyOrChainingValues(int inputLength)
    {
        byte[] key = HasherTests.MakeTestInput(32);
        byte[] input = HasherTests.MakeTestInput(inputLength);
        var hasher = Hasher.NewKeyed(key);
        hasher.Update(input);
        _ = hasher.Finalize();
        hasher.Dispose();

        // Everything secret (key, CV stack, chunk state) must be zero. What may remain is a few
        // bytes of non-secret header: flags, chunk base, stack length and the initialized flag.
        var bytes = MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref hasher, 1));
        int nonZero = 0;
        foreach (byte b in bytes) if (b != 0) nonZero++;
        Assert.True(nonZero <= 8, $"{nonZero} non-zero bytes remain after Dispose");
    }
}
