using System;
using System.Runtime.InteropServices;
using Blake3.Managed.Internal;

namespace Blake3.Managed.Tests;

public class Avx2DebugTests : Blake3TestsBase
{
    [Fact]
    public void TestAvx2TreeIntegration()
    {
        if (!HashManyAvx2.IsSupported)
        {
            return;
        }

        const string expected = "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63";

        var data = new byte[8192];
        for (int i = 0; i < data.Length; i++) data[i] = (byte)(i % 251);

        Span<uint> avx2Cvs = stackalloc uint[8 * 8];
        HashManyAvx2.HashMany(data, 8, Blake3Constants.IV, 0, 0, avx2Cvs);

        Span<uint> scalarCvs = stackalloc uint[8 * 8];
        for (int c = 0; c < 8; c++)
        {
            var chunkState = new Blake3Core.ChunkState(Blake3Constants.IV, (ulong)c, 0);
            chunkState.Update(data.AsSpan(c * 1024, 1024));
            var output = chunkState.CreateOutput();
            output.ChainingValue(scalarCvs.Slice(c * 8, 8));
        }

        for (int c = 0; c < 8; c++)
            for (int w = 0; w < 8; w++)
                Assert.True(scalarCvs[c * 8 + w] == avx2Cvs[c * 8 + w], $"Chunk {c}, CV word {w} mismatch");

        var publicHash = Hasher.Hash(data);
        AssertTextAreEqual(expected, publicHash.ToString());
    }

    [Fact]
    public void TestUpdate_LargeInputOnSmallStack_NoStackOverflow()
    {
        if (!HashManyAvx2.IsSupported)
        {
            return;
        }

        const int stackSize = 512 * 1024;
        const string expected = "7b64655021b6e77e4ca87cf26aade7feef862c87aa9c0277699fd77977b40dc0";

        Hash result = default;
        Exception? caught = null;

        var thread = new Thread(() =>
        {
            try { result = Hasher.Hash(HasherTests.MakeTestInput(20_000_000)); }
            catch (Exception ex) { caught = ex; }
        }, stackSize);

        thread.Start();
        thread.Join();

        Assert.True(caught == null, $"Exception on {stackSize / 1024} KB stack: {caught}");
        AssertTextAreEqual(expected, result.ToString());
    }
}
