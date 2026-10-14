using Blake3.Managed.Internal;

namespace Blake3.Managed.Tests;

[Collection(ParallelismCollection.Name)]
public class ArmJoinTests
{
    [Theory]
    [InlineData(32768)]
    [InlineData(32769)]
    [InlineData(65536)]
    [InlineData(131073)]
    [InlineData(1048576)]
    public void JoinedDeriveKeyAndSeekMatchSerialWithAnOccupiedSlot(int length)
    {
        byte[] data = HasherTests.MakeTestInput(length);
        byte[] expected = new byte[513];
        using var serial = Hasher.NewDeriveKey("ARM Join regression test");
        serial.Update(data);
        serial.Finalize(193, expected);

        foreach (bool occupied in new[] { false, true })
        {
            if (occupied) Blake3Tree.EnterMidSize();
            try
            {
                using var joined = Hasher.NewDeriveKey("ARM Join regression test");
                joined.UpdateWithJoin(data);
                joined.UpdateWithJoin(ReadOnlySpan<byte>.Empty);
                byte[] actual = new byte[513];
                joined.Finalize(193, actual);
                Assert.Equal(expected, actual);
            }
            finally
            {
                if (occupied) Blake3Tree.LeaveMidSize();
            }
        }
        int previous = Blake3Tree.EnterMidSize();
        Blake3Tree.LeaveMidSize();
        Assert.Equal(0, previous);
    }

    [Fact]
    public void ConcurrentJoinedCallsMatchSerialAndReleaseTheirSlots()
    {
        byte[] data = HasherTests.MakeTestInput(1024 * 1024 + 7);
        using var serial = Hasher.New();
        serial.Update(data);
        var expected = serial.Finalize();
        Parallel.For(0, 8, _ =>
        {
            using var joined = Hasher.New();
            joined.UpdateWithJoin(data);
            Assert.Equal(expected, joined.Finalize());
        });
        int previous = Blake3Tree.EnterMidSize();
        Blake3Tree.LeaveMidSize();
        Assert.Equal(0, previous);
    }
}
